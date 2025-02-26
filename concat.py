import PIL.Image as Image
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import resnet
from pathlib import Path
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
import vit
from tensorboardX import SummaryWriter
#from build_moat_cross import build_model


def pred(net,from_image):
    x=np.array(from_image).astype(np.float32)
    #print(x.shape)
    #x=x.transpose(2,1,0)
    x=np.expand_dims(x,0)
    x=torch.tensor(x)
    device = torch.device("cpu")
    net=net.to(device)
    x.to(device)
    #return net(x)
    predict_v = int(torch.max(net(x), dim=1)[1][0])
    return predict_v
"""
def convert_color(from_image,classes):
    #from_image=Image.fromarray(from_image.numpy())
    image=from_image.copy()
    r,g,b=image.split()
    if(classes==1):#high
        m=np.asarray(r)
        n=m.copy()
        n[n<255]=255
        r=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==3):#mid
        m=np.asarray(g)
        n=m.copy()
        n[n<255]=255
        g=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==2):#low
        m=np.asarray(b)
        n=m.copy()
        n[n<255]=255
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    else:
        image=from_image
    return image
#染全色
"""
def convert_color(from_image,classes):
    #from_image=Image.fromarray(from_image.numpy())
    image=from_image.copy()
    r,g,b=image.split()
    if(classes==1):#high
        m=np.asarray(r)
        n=m.copy()
        n[n<255]=255
        r=Image.fromarray(n)
        m=np.asarray(g)
        n=m.copy()
        n[n<256]=0
        g=Image.fromarray(n)
        m=np.asarray(b)
        n=m.copy()
        n[n<256]=0
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==3):#mid
        m=np.asarray(r)
        n=m.copy()
        n[n<256]=0
        r=Image.fromarray(n)
        m=np.asarray(g)
        n=m.copy()
        n[n<255]=255
        g=Image.fromarray(n)
        m=np.asarray(b)
        n=m.copy()
        n[n<256]=0
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==2):#low
        m=np.asarray(r)
        n=m.copy()
        n[n<256]=0
        r=Image.fromarray(n)
        m=np.asarray(g)
        n=m.copy()
        n[n<256]=0
        g=Image.fromarray(n)
        m=np.asarray(b)
        n=m.copy()
        n[n<255]=255
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==0):
        image=from_image
    return image


Image.MAX_IMAGE_PIXELS = None

IMAGES_PATH = 'D:/cancer/cancer_cut_test_256/'  # 图片集地址
IMAGES_FORMAT = ['.png', '.JPG']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 219//3  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 219//3  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'D:/cancer/result/final'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) < IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")
import numpy as np
#x=np.mat(np.zeros(13981712384).reshape(120832,115712))
#to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE//2, IMAGE_ROW * IMAGE_SIZE//2))
"""
net = vit.ViT(
        image_size = 1024,
        patch_size = 32,
        num_classes = 4,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1)
net=torch.load(r'D:\desktop\cut\vit\best.pth')
"""
def get_transform_for_train():
    transform_list = []
    #transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    #transform_list.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))#vit加了
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    fc=transforms.Compose(transform_list)
    return fc

fc=get_transform_for_train()

#net = resnet.resnet34()
#net = build_model()
net=torch.load(r'D:\cancer\cut\best_swin_batch32_nocj.pth')
#from_image = Image.open(IMAGES_PATH+'/'+str(38912)+'-'+str(38912)+'.png').convert('RGB')
#from_image = Image.open('D:/desktop/cancer_dataset_l/1/1/M53.png').convert('RGB')
#from_image_tensor=fc(from_image)
#print(pred(net,from_image_tensor))
#x=eval(input("stop"))


"""
full_dataset = torchvision.datasets.ImageFolder(root=r'D:\desktop\cancer_dataset_l\1', transform=get_transform_for_train())
train_size=int(0.8*len(full_dataset))
test_size=len(full_dataset)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


device = torch.device("cpu")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

for image, label in train_dataloader:
    image = image.to(device)
    label = label.to(device)
    output = net(image)
    predict_t = torch.max(output, dim=1)[1]
    print(predict_t)
"""




# 定义图像拼接函数
def image_compose(net,i,j):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(0, IMAGE_COLUMN):
        for x in range(0, IMAGE_ROW ):
            if(x==0):
                print(str(x*256)+'-'+str(y*256))
            from_image = Image.open(IMAGES_PATH+'/'+str(IMAGE_ROW*256*i+x*256)+'-'+str(IMAGE_COLUMN*256*j+y*256)+'.png').convert('RGB')
            from_image_tensor=fc(from_image)
            classes=pred(net,from_image_tensor)
            from_image = convert_color(from_image,classes)
            #from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                #(IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, ((x ) * IMAGE_SIZE, (y ) * IMAGE_SIZE))
    print("it's saving! please wait")
    return to_image.save(IMAGE_SAVE_PATH+str(i)+"-"+str(j)+".png") # 保存新图

#image_compose(net) #调用函数

for i in range(3):
    for j in range(3):
        image_compose(net,i,j)
