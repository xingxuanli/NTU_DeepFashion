import os
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# GLOBAL_PATH = 'drive/MyDrive/Colab Notebooks/DeepFashionProject/'
GLOBAL_PATH = ''

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

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=last_channel, out_features=512),
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

        classifier_in = 64
        # create sequential layers for all 6 cats
        self.cat1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_in, out_features=n_classes[0])
        )

        self.cat2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_in, out_features=n_classes[1])
        )
        
        self.cat3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_in, out_features=n_classes[2])
        )

        self.cat4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_in, out_features=n_classes[3])
        )

        self.cat5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_in, out_features=n_classes[4])
        )

        self.cat6 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_in, out_features=n_classes[5])
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        opt = {
            'cat1': self.cat1(x),
            'cat2': self.cat2(x),
            'cat3': self.cat3(x),
            'cat4': self.cat4(x),
            'cat5': self.cat5(x),
            'cat6': self.cat6(x)
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


def result_matrics(nn_output, ground_truth):
    nn_output_label = {}
    accuracy = {}
    for cat in nn_output:
        _, tmp_predicted = nn_output[cat].cpu().max(1)
        tmp_groundtruth = ground_truth[cat].cpu()
        nn_output_label[cat] = [tmp_predicted, tmp_groundtruth]

    # ignore warning
    with warnings.catch_warnings():
         warnings.simplefilter('ignore')
         for cat in nn_output_label:
             accuracy[cat] = accuracy_score(y_true=nn_output_label[cat][1].numpy(),
                                            y_pred=nn_output_label[cat][0].numpy())
             
    return accuracy


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

def validate(model, dataloader, device, logger=None, epoch=None, checkpoint=None):
    if checkpoint is not None:
       checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
         avg_loss = 0
         val_accuracies = {
             'cat1': 0,
             'cat2': 0,
             'cat3': 0,
             'cat4': 0,
             'cat5': 0,
             'cat6': 0,
         }

         for batch in dataloader:
             img = batch['img'].to(device)
             ground_truth = batch['labels']
             ground_truth = {k: ground_truth[k].to(device) for k in ground_truth}
             nn_output = model(img)
            
             loss, loss_each = loss_function(nn_output, ground_truth)
             avg_loss += loss.item()
             accuracy = result_matrics(nn_output, ground_truth)
             
             for cat in val_accuracies:
                 val_accuracies[cat] += accuracy[cat]

    n_batches = len(dataloader)
    # batch loss?
    avg_loss /= n_batches
    for cat in val_accuracies:
        val_accuracies[cat] /= n_batches
    print('-' * 77)
    print("Validation loss: {:.4f}, Cat1: {:.4f}, Cat2: {:.4f}, Cat3: {:.4f}, Cat4: {:.4f}, Cat5: {:.4f}, Cat6: {:.4f}, avg: {:.4f}\n".format(
        avg_loss, val_accuracies['cat1'], val_accuracies['cat2'], val_accuracies['cat3'],
        val_accuracies['cat4'], val_accuracies['cat5'], val_accuracies['cat6'], sum(val_accuracies.values())/6 
    ))

    # log the info
    if logger is not None and epoch is not None:
       logger.add_scalar('val_loss', avg_loss, epoch)
       for cat in val_accuracies:
           logger.add_scalar('val_accuracy/'+cat, val_accuracies[cat], epoch)
       logger.add_scalar('val_accuracy/avg', sum(val_accuracies.values())/6, epoch)

    return avg_loss, val_accuracies


def get_cur_time():
    timezone_offset = 8.0
    tzinfo = timezone(timedelta(hours=timezone_offset))
    return datetime.strftime(datetime.now(tzinfo), '%Y-%m-%d_%H-%M')

def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)

    return f


def train(start_epoch=1, n_epochs=50, batch_size=32, num_workers=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_PATH = os.path.join(GLOBAL_PATH, 'FashionDataset/')
    SPLIT_PATH = os.path.join(GLOBAL_PATH, 'FashionDataset/split/')

    # train data
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = FashionDataset(img_path=IMG_PATH, split_path=SPLIT_PATH, 
                                transform = train_transform, flag='train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)

    # val data
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_data = FashionDataset(img_path=IMG_PATH, split_path=SPLIT_PATH, 
                                transform = val_transform, flag='val')
    val_dataloader = DataLoader(val_data, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)


    # total_loss_list = []
    # average_loss_list = []

    model = MultiLabelModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # log stuff
    logdir = os.path.join(GLOBAL_PATH, 'logs', get_cur_time())
    print(logdir)
    savedir = os.path.join(GLOBAL_PATH, 'checkpoints', get_cur_time())
    print(savedir)
    os.makedirs(logdir, exist_ok=False)
    os.makedirs(savedir, exist_ok=False)
    logger = SummaryWriter(logdir)

    print('Start training ...')

    model.train()
    checkpoint_path = None
    all_loss_accuracy = {}
    n_train_batches = len(train_dataloader)
    for epoch in range(start_epoch, n_epochs+1):
        total_loss = 0
        train_accuracies = {
             'cat1': 0,
             'cat2': 0,
             'cat3': 0,
             'cat4': 0,
             'cat5': 0,
             'cat6': 0,
         }
        
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            img = batch['img'].to(device)
            ground_truth = batch['labels']
            ground_truth = {k: ground_truth[k].to(device) for k in ground_truth}
            nn_output = model(img)
            
            loss, loss_each = loss_function(nn_output, ground_truth)
            total_loss += loss.item()
            accuracy = result_matrics(nn_output, ground_truth)
             
            for cat in train_accuracies:
                train_accuracies[cat] += accuracy[cat]
            
            loss.backward()
            optimizer.step()

        n_train_batches = len(train_dataloader)

        # batch loss?
        total_loss /= n_train_batches
        for cat in train_accuracies:
            train_accuracies[cat] /= n_train_batches
        print("epoch {:4d}, loss: {:.4f}, Cat1: {:.4f}, Cat2: {:.4f}, Cat3: {:.4f}, Cat4: {:.4f}, Cat5: {:.4f}, Cat6: {:.4f}, avg: {:.4f}\n".format(
            epoch,
            total_loss, train_accuracies['cat1'], train_accuracies['cat2'], train_accuracies['cat3'],
            train_accuracies['cat4'], train_accuracies['cat5'], train_accuracies['cat6'], sum(train_accuracies.values())/6
        ))

        logger.add_scalar('train_loss', total_loss/n_train_batches, epoch)
        for cat in train_accuracies:
           logger.add_scalar('train_accuracy/'+cat, train_accuracies[cat], epoch)
        logger.add_scalar('train_accuracy/avg', sum(train_accuracies.values())/6, epoch)

        if epoch % 10 == 0:
           checkpoint_path = checkpoint_save(model, savedir, epoch)

        if epoch % 1 == 0:
           val_avg_loss, val_accuracies = validate(model, val_dataloader, device, logger, epoch)
        
        all_loss_accuracy[epoch] = {'train_loss': total_loss, 'train_accuracy': train_accuracies,
                                    'val_loss': val_avg_loss, 'val_accuracy': val_accuracies}
    checkpoint_path = checkpoint_save(model, savedir, epoch - 1)
    return checkpoint_path, all_loss_accuracy


last_checkpoint_path, all_loss_accuracy = train(start_epoch=1, n_epochs=100, batch_size=32, num_workers=2)

with open('final_opt', 'w') as f:
    f.write(all_loss_accuracy)
f.close()