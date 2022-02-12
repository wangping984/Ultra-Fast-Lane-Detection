# %% [markdown]
# 分析test.py文件

# %%
import torch
from model.model import parsingNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from PIL import Image
import cv2
import scipy.special

torch.backends.cudnn.benchmark = True
cls_num_per_lane = 18
net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,cls_num_per_lane,4),
                    use_aux=False).cpu()

modlePath = 'culane_18.pth'
state_dict = torch.load(modlePath, map_location = 'cpu')['model']


# %%
net.load_state_dict(state_dict, strict = False)
net.eval()

# %%
img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

cap = cv2.VideoCapture("20190408035014_020328AA.MP4")
_,img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = Image.fromarray(img)
x = img_transforms(img2)
x = x.unsqueeze(0).cpu()+1

# %%

img_path = "mytest.jpg"
image = Image.open(img_path)
img = img_transforms(image)
img = img.cpu()
img = img.unsqueeze(0).cpu()+1
with torch.no_grad():
    out = net(img)

# %% [markdown]
# 下面参照demo.py处理输出数据

# %%
out_j = out[0].data.cpu().numpy()
# 下面让18行row ankor上下颠倒排列
out_j = out_j[:, ::-1, :]
# softmax的参数axis=0，表示只对201个gridding做softmax运算
# out_j1[:-1, :, :]表示第一维度gridding数量减1，去掉最后一个
prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

# %%
import numpy as np
idx = np.arange(200) + 1
idx1 = idx.reshape(-1, 1, 1)

loc = np.sum(prob * idx1, axis=0)
out_j = np.argmax(out_j, axis=0)
loc[out_j == 200] = 0
out_j = loc

# %%
vis = cv2.imread(img_path)
col_sample = np.linspace(0, 800 - 1, 200)
col_sample_w = col_sample[1] - col_sample[0]
img_w, img_h = 1640, 590
row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
for i in range(out_j.shape[1]):
    if np.sum(out_j[:, i] != 0) > 2:
        for k in range(out_j.shape[0]):
            if out_j[k, i] > 0:
                ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                cv2.circle(vis,ppp,5,(0,255,0),-1)

# %%
cv2.imwrite('out4.jpg', vis)



