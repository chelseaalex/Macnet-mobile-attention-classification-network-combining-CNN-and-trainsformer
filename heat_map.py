import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
from build_moat_cross_feature import build_model
import os
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

if __name__ == "__main__":
    device = torch.device("cuda")
    model = build_model()
    model = torch.load("D:/desktop/cut/best_moatcrossfeature_32_d2372.pth")
    model.to('cuda')
    img_path="D:/desktop/cancer_dataset_256/low/L106.png"
    img = Image.open(img_path).convert('RGB')
    #img = img[:, :, ::-1] # BGR to RGB.
    """
    # to PIL.Image
    img = Image.fromarray(img)
    img = img_transforms(img)"""
    img = fc(img)
    x=np.array(img).astype(np.float32)
    x=np.expand_dims(x,0)
    print(x.shape)
    img=torch.tensor(x)
    img = img.to('cuda')


    output,feature_map,classifier_weight = model(img)
    print(output.shape)
    print(classifier_weight.shape)
    x=output[0].tolist()
    pred=x.index(max(x))
    #print(feature_map)
    #test()
    
    class_ = {0:'back', 1:'high', 2:'low', 3:'mid'}
    def returnCAM(feature_conv, weight_softmax, idx):
        bz, nc, h, w = feature_conv.shape        #1,960,7,7
        print(bz, nc, h, w)
        output_cam = []
        
        feature_conv = feature_conv.reshape((nc, h*w))  # [960,7*7]
        print(feature_conv.shape)
        cam = weight_softmax[idx].unsqueeze(0).matmul(feature_conv)  #(5, 960) * (960, 7*7) -> (5, 7*7) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.to("cpu")
        cam_img = cam_img.detach().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)
     
            #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
        return output_cam
    # CAMs = returnCAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM
    CAMs = returnCAM(feature_map[-1], classifier_weight, pred)
    
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result=heatmap*0.5+img*0.5
    CAM_RESULT_PATH = r'D:/desktop/cut/result/'   #CAM结果的存储地址
    if not os.path.exists(CAM_RESULT_PATH):
        os.mkdir(CAM_RESULT_PATH)
    image_name_ = img_path[-6:-4]
    cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + str(pred) + '.png', result)

    """

    

    img = Image.open("D:/desktop/cancer_dataset_256/high/H1.png")
    
    # get heatmap and original image
    heatmap = get_heat(model, img)
    
    rgb_img = Image.open("D:/desktop/cancer_dataset_256/high/H1.png").convert('RGB')
    rgb_img = Image.resize(rgb_img, (256, 256))
    pad_size = (256 - 256) // 2
    rgb_img = rgb_img[pad_size:-pad_size, pad_size:-pad_size]

    mix = rgb_img * 0.5 + heatmap * 0.5
    mix = mix.astype(np.uint8)

    # cv2.namedWindow('heatmap', 0)
    # cv2.imshow('heatmap', heatmap)
    # cv2.namedWindow('rgb_img', 0)
    # cv2.imshow('rgb_img', rgb_img)
    # cv2.namedWindow('mix', 0)
    # cv2.imshow('mix', mix)
    # cv2.watiKey(0)
    mix.save(IMAGE_SAVE_PATH+str(i)+"-"+str(j)+".png")"""
    """if args.save_img != "":
        cv2.imwrite(args.save_img + "/heatmap.jpg", heatmap)
        cv2.imwrite(args.save_img + "/rbg_img.jpg", rgb_img)
        cv2.imwrite(args.save_img + "/mix.jpg", mix)"""
