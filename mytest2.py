import torch
from model.model import parsingNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from PIL import Image
import cv2
import scipy.special
import os
import numpy as np



use_gpu = False

def netInit():

    torch.backends.cudnn.benchmark = True
    cls_num_per_lane = 18
    modlePath = 'culane_18.pth'
    if use_gpu == True:
        net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,cls_num_per_lane,4),
                            use_aux=False).cuda()
        state_dict = torch.load(modlePath, map_location = 'cuda')['model']
    else:
        net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,cls_num_per_lane,4),
                            use_aux=False).cpu()
        state_dict = torch.load(modlePath, map_location = 'cpu')['model']

    net.load_state_dict(state_dict, strict = False)
    net.eval()
    return net

def img_trans(input_img):
    img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])    
    img = img_transforms(input_img)
    if use_gpu == True:
        img = img.cuda() 
        img = img.unsqueeze(0).cuda()+1
    else:
        img = img.cpu() 
        img = img.unsqueeze(0).cpu()+1
    return img

net = netInit()

# somedir = 'C:\\Users\\wp\\Downloads\\driver_37_30frame.tar\\driver_37_30frame\\05181520_0219.MP4'
somedir = os.getcwd()
files = [f for f in os.listdir(somedir) 
    if os.path.isfile(os.path.join(somedir, f)) 
    and f.endswith(".jpg")]
# files = [ fi for fi in files if fi.endswith(".jpg") ]

col_sample = np.linspace(0, 800 - 1, 200)
col_sample_w = col_sample[1] - col_sample[0]
row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
cls_num_per_lane = 18

results_path = somedir

with torch.no_grad():
    for f in files:
        # print(f)
        img_path = os.path.join(somedir, f)
        img = Image.open(img_path)
        

        out = net(img_trans(img))
        out_j = out[0].data.cpu().numpy()
        # 下面让18行row ankor上下颠倒排列
        out_j = out_j[:, ::-1, :]
        # softmax的参数axis=0，表示只对201个gridding做softmax运算
        # out_j1[:-1, :, :]表示第一维度gridding数量减1，去掉最后一个
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(200) + 1
        idx1 = idx.reshape(-1, 1, 1)

        loc = np.sum(prob * idx1, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == 200] = 0
        out_j = loc

        img_w, img_h = img.size

        vis = cv2.imread(img_path)
        
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,5,(0,255,0),-1)
        
        filename = os.path.join(results_path, 're_'+f)
        cv2.imwrite(filename, vis)





