import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader



class FashionDataset(Dataset):
    def __init__(self, img_path='FashionDataset/', 
                 split_path='FashionDataset/split/', 
                 transform=None, flag=None):

        super().__init__()
        
        self.data = []
        self.labels = []
        self.bbox = []
        self.transform = transform
        self.flag = flag
        
        if flag == 'train':
            X_path = os.path.join(split_path, 'train.txt')
            X_files = open(X_path).read().split('\n')
            y_path = os.path.join(split_path, 'train_attr.txt')
            y_files = open(y_path).read().split('\n')

            bbox_path = os.path.join(split_path, 'train_bbox.txt')
            bbox_files = open(bbox_path).read().split('\n')
            
        elif flag == 'val':
            X_path = os.path.join(split_path, 'val.txt')
            X_files = open(X_path).read().split('\n')
            y_path = os.path.join(split_path, 'val_attr.txt')
            y_files = open(y_path).read().split('\n')

            bbox_path = os.path.join(split_path, 'val_bbox.txt')
            bbox_files = open(bbox_path).read().split('\n')

        elif flag == 'test':
            X_path = os.path.join(split_path, 'test.txt')
            X_files = open(X_path).read().split('\n')

            bbox_path = os.path.join(split_path, 'test_bbox.txt')
            bbox_files = open(bbox_path).read().split('\n')

        else:
            raise ValueError('Invalid flag value, should be train/val/test')
            
        for i in range(len(X_files)):
            # images path
            self.data.append(os.path.join(img_path, X_files[i]))

            # bounding box
            self.bbox.append([int(x) for x in bbox_files[i].split(' ')])

            if flag != 'test':
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

        # crop by bounding box
        img = img.crop((self.bbox[idx][0], self.bbox[idx][1],
                        self.bbox[idx][2], self.bbox[idx][3]))
        
        # check if transform
        if self.transform:
            img = self.transform(img)
        
        if self.flag != 'test':
            opt_data = {
                'img': img,
                'labels': self.labels[idx]
            }
            return opt_data
        
        else:
            opt_data = {
                'img': img
            }
            return opt_data
    
    def __len__(self):
        return len(self.data)