import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from focalloss import FocalLoss
import pretrainedmodels



class MultiLabelModel(nn.Module):
    def __init__(self, n_classes=[7, 3, 3, 4, 6, 3]):
        super().__init__()
        # pretrained resnet50 as base model
        # self.resnet50 = models.resnet50(pretrained=True)
        # self.resnet50.fc = nn.Linear(in_features=2048, out_features=1024)
        # self.model = models.resnet50(pretrained=True)
        # self.model = pretrainedmodels.se_resnet101(num_classes=1000, pretrained='imagenet')
        self.model = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        # self.model = pretrainedmodels.nasnetalarge(num_classes=1000, pretrained='imagenet')

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU()
        )

        classifier_in = 1000
        # create sequential layers for all 6 cats
        self.cat1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=classifier_in, out_features=n_classes[0])
        )

        self.cat2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=classifier_in, out_features=n_classes[1])
        )
        
        self.cat3 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=classifier_in, out_features=n_classes[2])
        )

        self.cat4 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=classifier_in, out_features=n_classes[3])
        )

        self.cat5 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=classifier_in, out_features=n_classes[4])
        )

        self.cat6 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=classifier_in, out_features=n_classes[5])
        )

    def forward(self, x):
        x = self.model(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        
        opt = {
            'cat1': self.cat1(x),
            'cat2': self.cat2(x),
            'cat3': self.cat3(x),
            'cat4': self.cat4(x),
            'cat5': self.cat5(x),
            'cat6': self.cat6(x)
        }
        
        return opt

    def loss_function(self, nn_output, ground_truth):
        sum_loss = 0
        opt = {}

        weights = {
            'cat1': 0.95,#2
            'cat2': 0.85,#4
            'cat3': 0.85,#4
            'cat4': 0.9,#3
            'cat5': 1,#1
            'cat6': 0.9,#3
        }
        
        for cat in nn_output:
            # tmp_loss = F.cross_entropy(nn_output[cat], ground_truth[cat])

            # use focal loss
            num_class = nn_output[cat].size(1)
            focal_loss = FocalLoss(num_class)
            tmp_loss = focal_loss(nn_output[cat], ground_truth[cat])

            opt[cat] = weights[cat]*tmp_loss
            sum_loss += weights[cat]*tmp_loss
            
        return sum_loss, opt
