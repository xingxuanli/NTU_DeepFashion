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


time = '2021-09-22_16-36'
e = '70'

GLOBAL_PATH = ''
batch_size = 32
num_workers = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt_list = []

model_path = 'checkpoints/'+time+'/checkpoint-0000'+e+'.pth'
model = MultiLabelModel()
model.load_state_dict(torch.load(model_path))
model = model.to(device)

IMG_PATH = os.path.join(GLOBAL_PATH, 'FashionDataset/')
SPLIT_PATH = os.path.join(GLOBAL_PATH, 'FashionDataset/split/')

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = FashionDataset(img_path=IMG_PATH, split_path=SPLIT_PATH, 
                            transform = test_transform, flag='test')
test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)

final_opt = []
with torch.no_grad():
    model.eval()

    for batch in test_dataloader:
        img = batch['img'].to(device)
        nn_output = model(img)
        
        tmp_label = []
        for cat in nn_output:
            _, tmp_predicted = nn_output[cat].cpu().max(1)
            tmp_label.append(tmp_predicted.numpy())
            
        final_opt.append(np.array(tmp_label).T)
final_array = np.concatenate(final_opt)

with open('prediction.txt', 'w') as f: 
    for i in range(final_array.shape[0]):
        for j in range(5):
            f.write(str(final_array[i][j]))
            f.write(' ')
        f.write(str(final_array[i][5]))
        f.write('\n')

import zipfile
zipfile.ZipFile('prediction.zip', mode='w').write('prediction.txt')