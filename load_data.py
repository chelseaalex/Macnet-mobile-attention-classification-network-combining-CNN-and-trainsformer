import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import torchvision
from torchvision import transforms
import torch


def get_transform_for_train():
    transform_list = []
    #transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    transform_list.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)



if __name__ == '__main__':
    full_dataset = torchvision.datasets.ImageFolder(root=r'D:\desktop\cancer_dataset', transform=get_transform_for_train())
    train_size=int(0.8*len(full_dataset))
    test_size=len(full_dataset)-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    #for index, batch_data in enumerate(dataloader):
    #    print(index, batch_data[0].shape, batch_data[1].shape)
    
