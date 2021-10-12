import os
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

from model import MultiLabelModel
from dataloader import FashionDataset
from utils import result_matrics, checkpoint_save, checkpoint_load, get_cur_time



def validate(model, dataloader, device, logger=None, epoch=None, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_loss_each = {
            'cat1': 0, 'cat2': 0, 'cat3': 0,
            'cat4': 0, 'cat5': 0, 'cat6': 0
        }

        val_accuracies = {
            'cat1': 0, 'cat2': 0, 'cat3': 0,
            'cat4': 0, 'cat5': 0, 'cat6': 0
        }

        for batch in dataloader:
            img = batch['img'].to(device)
            ground_truth = batch['labels']
            ground_truth = {k: ground_truth[k].to(device) for k in ground_truth}
            nn_output = model(img)
            loss, loss_each = model.loss_function(nn_output, ground_truth)
            
            # log the loss and accuracies
            val_loss += loss.item()
            accuracy = result_matrics(nn_output, ground_truth)
            val_loss_each = {k: val_loss_each[k]+loss_each[k] for k in val_loss_each}
            val_accuracies = {k: val_accuracies[k]+accuracy[k] for k in val_accuracies}


    n_batches = len(dataloader)

    # batch loss and avg accuracies
    val_loss /= n_batches
    val_accuracies = {k: val_accuracies[k]/n_batches for k in val_accuracies}

    # display info in terminal
    print('-' * 77)
    print("""Validation loss: {:.4f}, 
            Cat1: {:.4f}, Cat2: {:.4f}, Cat3: {:.4f}, 
            Cat4: {:.4f}, Cat5: {:.4f}, Cat6: {:.4f}, 
            avg: {:.4f}\n""".format(
                                    val_loss, 
                                    val_accuracies['cat1'], val_accuracies['cat2'], val_accuracies['cat3'], 
                                    val_accuracies['cat4'], val_accuracies['cat5'], val_accuracies['cat6'], 
                                    sum(val_accuracies.values())/6
        ))

    # log the info
    if logger is not None and epoch is not None:
        logger.add_scalar('val_loss/avg', val_loss, epoch)
        logger.add_scalar('val_accuracy/avg', sum(val_accuracies.values())/6, epoch)
        for cat in val_accuracies:
            logger.add_scalar('val_loss/'+cat, val_loss_each[cat], epoch)
            logger.add_scalar('val_accuracy/'+cat, val_accuracies[cat], epoch)

    # set model back to train
    model.train()
    return val_loss, val_accuracies


def train(model, optimizer, train_dataloader, val_dataloader, device, 
            n_epochs=50, logger=None, savedir=None, f_model=10, f_val=1):
    best_val_accuracy = 0
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(1, n_epochs):

        train_loss = 0
        train_loss_each = {
            'cat1': 0, 'cat2': 0, 'cat3': 0,
            'cat4': 0, 'cat5': 0, 'cat6': 0
        }

        train_accuracies = {
            'cat1': 0, 'cat2': 0, 'cat3': 0,
            'cat4': 0, 'cat5': 0, 'cat6': 0
        }

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            img = batch['img'].to(device)
            ground_truth = batch['labels']
            ground_truth = {k: ground_truth[k].to(device) for k in ground_truth}
            nn_output = model(img)
            loss, loss_each = model.loss_function(nn_output, ground_truth)
            
            # log the loss and accuracies
            train_loss += loss.item()
            accuracy = result_matrics(nn_output, ground_truth)
            train_loss_each = {k: train_loss_each[k]+loss_each[k] for k in train_loss_each}
            train_accuracies = {k: train_accuracies[k]+accuracy[k] for k in train_accuracies}

            loss.backward()
            optimizer.step()
        
        # scheduler.step()
        print('learning rate:', optimizer.param_groups[0]['lr'])
        n_train_batches = len(train_dataloader)

        # batch loss and avg accuracies
        train_loss /= n_train_batches
        train_accuracies = {k: train_accuracies[k]/n_train_batches for k in train_accuracies}

        # display info in terminal
        print("""epoch {:4d}, loss: {:.4f}, 
                Cat1: {:.4f}, Cat2: {:.4f}, Cat3: {:.4f}, 
                Cat4: {:.4f}, Cat5: {:.4f}, Cat6: {:.4f}, 
                avg: {:.4f}\n""".format(
                                        epoch, train_loss, 
                                        train_accuracies['cat1'], train_accuracies['cat2'], train_accuracies['cat3'], 
                                        train_accuracies['cat4'], train_accuracies['cat5'], train_accuracies['cat6'], 
                                        sum(train_accuracies.values())/6
            ))

        # log the info
        if logger is not None and epoch is not None:
            logger.add_scalar('train_loss/avg', train_loss, epoch)
            logger.add_scalar('train_accuracy/avg', sum(train_accuracies.values())/6, epoch)
            for cat in train_accuracies:
                logger.add_scalar('train_loss/'+cat, train_loss_each[cat], epoch)
                logger.add_scalar('train_accuracy/'+cat, train_accuracies[cat], epoch)

        # model weights saving
        if epoch % f_model == 0:
            checkpoint_path = checkpoint_save(model, savedir, epoch)
        if epoch % f_val == 0:
            val_loss, val_accuracies = validate(model, val_dataloader, device, logger, epoch)
            avg_val_accuracy = sum(val_accuracies.values())/6
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print('#*#*#*#*#*#*#*#*#*# best model saved at epoch', epoch)




def main(batch_size, num_workers, n_epochs, f_model, f_val):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = MultiLabelModel().to(device)
    #model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    ### train and val dataloader
    GLOBAL_PATH = ''
    IMG_PATH = os.path.join(GLOBAL_PATH, 'FashionDataset/')
    SPLIT_PATH = os.path.join(GLOBAL_PATH, 'FashionDataset/split/')

    # train
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = FashionDataset(img_path=IMG_PATH, split_path=SPLIT_PATH, 
                                transform = train_transform, flag='train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)

    # val 
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_data = FashionDataset(img_path=IMG_PATH, split_path=SPLIT_PATH, 
                                transform = val_transform, flag='val')
    val_dataloader = DataLoader(val_data, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)


    ### log address
    logdir = os.path.join(GLOBAL_PATH, 'logs', get_cur_time())
    print(logdir)
    savedir = os.path.join(GLOBAL_PATH, 'checkpoints', get_cur_time())
    print(savedir)
    os.makedirs(logdir, exist_ok=False)
    os.makedirs(savedir, exist_ok=False)
    logger = SummaryWriter(logdir)


    ### run the training
    print('Start training ...')
    train(model, optimizer, train_dataloader, val_dataloader, device, 
          n_epochs=n_epochs, logger=logger, savedir=savedir, f_model=f_model, f_val=f_val)


batch_size = 32
num_workers = 8
n_epochs = 30
f_model = 10
f_val = 1
main(batch_size, num_workers, n_epochs, f_model, f_val)
