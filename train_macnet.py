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
#from build_moat import build_model
#from build_moat_cross import build_model
#from build_moat_cross_feature import build_model
from build_macnet import build_model
writer1=SummaryWriter(r'tensorboard/multimoat55_2372')

from thop import profile

def get_transform_for_train():
    transform_list = []
    #transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    #transform_list.append(transforms.ColorJitter(0.5, 0.5, 0.5, 0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

def train_and_val(epochs, model, train_loader, val_loader, criterion, optimizer):
    torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:
                # training phase

                #                 images, labels = data
                #             optimizer.zero_grad()
                #             logits = net(images.to(device))
                #             loss = loss_function(logits, labels.to(device))
                #             loss.backward()
                #             optimizer.step()

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                # forward
                output = model(image)
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]

                # backward
                loss.backward()
                optimizer.step()  # update weight

                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()
                pbar.update(1)

        model.eval()
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]

                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculatio mean for each batch
            train_loss.append(running_loss / len(train_dataset))
            val_loss.append(val_losses / len(test_dataset))

            train_acc.append(training_acc / len(train_dataset))
            val_acc.append(validation_acc / len(test_dataset))
            
            torch.save(model, "last_multimoat55_2372.pth")
            if best_acc<(validation_acc / len(test_dataset)):
                torch.save(model, "best_multimoat55_2372.pth")
            
            writer1.add_scalar('Train_Acc',(training_acc / len(train_dataset)),global_step=e)
            writer1.add_scalar('Train Loss',(running_loss / len(train_dataset)),global_step=e)
            writer1.add_scalar('Val Acc',(validation_acc / len(test_dataset)),global_step=e)
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Acc: {:.3f}..".format(training_acc / len(train_dataset)),
                  "Val Acc: {:.3f}..".format(validation_acc / len(test_dataset)),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_dataset)),
                  "Val Loss: {:.3f}..".format(val_losses / len(test_dataset)),
                  "Time: {:.2f}s".format((time.time() - since)))
            

    history = {'train_loss': train_loss, 'val_loss': val_loss,'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    
    return history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = torchvision.datasets.ImageFolder(root=r'D:\desktop\cancer_dataset_256', transform=get_transform_for_train())
    train_size=int(0.8*len(full_dataset))#0.8
    test_size=len(full_dataset)-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    """train_size=int(0.2*len(full_dataset))
    test_size=len(full_dataset)-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_size=int(0.8*len(train_dataset))
    test_size=len(train_dataset)-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    print(len(train_dataset))"""
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    net = build_model()
    x=torch.randn(1,3,256,256)
    flops, params =profile(net, inputs=(x,))
    print(flops)
    print(params)
    #net = resnet.resnet34()
    #x=input("1")
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
    """
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.AdamW(net.parameters(), lr=0.001)  # 设置优化器和学习率
    epoch = 300
    history = train_and_val(epoch, net, train_dataloader, test_dataloader, loss_function, optimizer)


