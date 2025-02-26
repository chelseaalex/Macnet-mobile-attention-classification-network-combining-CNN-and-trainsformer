from PIL import Image
import os
import shutil
import glob
import cv2
import numpy as np, pandas as pd
from collections import Counter
path = 'D:/desktop/cancer_dataset/back'
new_path = 'D:/desktop/cancer_dataset_256/back'
k=0
img_size=1024
target_size=256
"""
def judge(i):
    x=Counter(list(i.getdata())).most_common(150)
    print(x)
    if(len(x)==200):
        x1=x[199][0]
        x2=x[198][0]
        x3=x[197][0]
        x4=x[196][0]
        x5=x[195][0]
        print(x1,x2,x3,x4,x5)
        if np.mean(x1)>200 and np.mean(x2)>200 and np.mean(x3)>200 and np.mean(x4)>200 and np.mean(x5)>200:  
            return 0
        else:
            return 1
    else:
        return 0
"""
import cv2
def judge(img):
    #img=cv2.imread('D:/desktop/cancer_dataset/low/L1798.png')
    h,w,_=img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    w=len(img[img==255])
    b=len(img[img==0])
    if b>w:
        return 1
    else:
        return 0



for root,dirs,files in os.walk(path):
    for i in range(len(files)):
        if files[i][-3:] == 'png' or files[i][-3:] == 'PNG':
            file_path = root + '/' + files[i]
            img = Image.open(file_path).convert('RGB')
            #img2=cv2.imread(file_path)
            for o in range(img_size//target_size):
                for p in range(img_size//target_size):
                    k=k+1
                    region = img.crop((o*256,p*256,(o+1)*256,(p+1)*256))
                    arr_region=np.asarray(region)
                    region.save(new_path+'/' + 'B'+str(k) + '.png')
                    #if(judge(arr_region)):
                    #    region.save(new_path+'/' + 'B'+str(k) + '.png')
                    #print(k)
                    #region.save(new_path+'/' + 'B'+str(k) + '.png')


#i=Image.open('D:/desktop/cancer_dataset_256/low/L1468.png').convert('RGB')
#i=arr_region=np.asarray(i)
#print(judge(i))
