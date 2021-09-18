import os
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class FashionDataset(Dataset):
    def __init__(self, img_path='FashionDataset/', 
                 split_path='FashionDataset/split/', 
                 transform=None, flag=None):

        super().__init__()
        
        self.data = []
        self.labels = []
        self.transform = transform
        
        if flag == 'train':
            X_path = os.path.join(split_path, 'train.txt')
            X_files = open(X_path).read().split('\n')
            y_path = os.path.join(split_path, 'train_attr.txt')
            y_files = open(y_path).read().split('\n')
            
        if flag == 'val':
            X_path = os.path.join(split_path, 'val.txt')
            X_files = open(X_path).read().split('\n')
            y_path = os.path.join(split_path, 'val_attr.txt')
            y_files = open(y_path).read().split('\n')
            
        for i in range(len(X_files)):
            # images path
            self.data.append(os.path.join(img_path, X_files[i]))

            # labels
            tmp_labels = y_files[i].split(' ')
            self.labels.append({
                'cat1': int(tmp_labels[0]),
                'cat2': int(tmp_labels[1]),
                'cat3': int(tmp_labels[2]),
                'cat4': int(tmp_labels[3]),
                'cat5': int(tmp_labels[4]),
                'cat6': int(tmp_labels[5])
            })
            
    def __getitem__(self, idx):
        # read image
        img_path = self.data[idx]
        img = Image.open(img_path)
        
        # check if transform
        if self.transform:
            img = self.transform(img)
            
        opt_data = {
            'img': img,
            'labels': self.labels[idx]
        }
        return opt_data
    
    def __len__(self):
        return len(self.data)


class MultiLabelModel(nn.Module):
    def __init__(self, n_classes=[7, 3, 3, 4, 6, 3]):
        super().__init__()
        # pretrained resnet50 as base model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # size of last channel before classifier
        last_channel = models.resnet50().fc.out_features
        
        # create sequential layers for all 6 cats
        self.cats = []
        for i in range(len(n_classes)):
            tmp_cat = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=last_channel, out_features=n_classes[i])
            )
            self.cats.append(tmp_cat)
            
    def forward(self, x):
        x = self.resnet50(x)
        
        opt = {
            'cat1': self.cats[0](x),
            'cat2': self.cats[1](x),
            'cat3': self.cats[2](x),
            'cat4': self.cats[3](x),
            'cat5': self.cats[4](x),
            'cat6': self.cats[5](x)
        }
        
        return opt


def loss_function(nn_output, ground_truth):
    sum_loss = 0
    opt = {}
    
    for cat in nn_output:
        tmp_loss = F.cross_entropy(nn_output[cat], ground_truth[cat])
        opt[cat] = tmp_loss
        sum_loss += tmp_loss
        return sum_loss, opt


device = torch.device('cpu')
batch_size = 64
torch.set_num_threads = 2
num_workers = 0

# train data
flag = 'train'
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = FashionDataset(transform = train_transform, flag=flag)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = MultiLabelModel()
optimizer = torch.optim.Adam(model.parameters())

model.train
for epoch in range(100):
    total_loss = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        print(1)
        img = batch['img']
        ground_truth = batch['labels']
        nn_output = model(img)
        
        print(2)
        loss, loss_each = loss_function(nn_output, ground_truth)
        total_loss += loss.item()
        
        print(3)
        loss.backward()
        optimizer.step()
        
    print('Epoch', epoch, 'loss', total_loss)