import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


train_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),  #p=probability, which means 50% chances to flip the image
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])


def create_dataloaders(train_dir, test_dir, batch_size, num_workers=0) :

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform = train_transform,
                                      target_transform = None)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform = test_transform)
    
    class_names = train_data.classes
    #class_dict = train_data.class_to_idx

    train_dataloader = DataLoader(dataset = train_data,
                                  batch_size = batch_size,
                                  num_workers = num_workers,
                                  shuffle = True)

    test_dataloader = DataLoader(dataset = test_data,
                                batch_size = batch_size,
                                num_workers = num_workers,
                                shuffle = False)
    
    return train_dataloader, test_dataloader, class_names