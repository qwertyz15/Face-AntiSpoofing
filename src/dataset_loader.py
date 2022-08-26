# Original code https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# Author : @zhuyingSeu , Company : Minivision
# Modified by @hairymax

import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd

#from src.data_io import transform as trans

def opencv_loader(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg

class CelebADataset(Dataset):
    def __init__(self, root, labels, transform=None, target_transform=None,
                 loader=opencv_loader):
        self.root = root
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.labels.iloc[idx, 0])
        sample = self.loader(path)
        target = self.labels.iloc[idx, 1]
        
        if sample is None:
            print('image is None --> ', path)
        assert sample is not None
        
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class CelebADatasetFT(CelebADataset):
    def __init__(self, root, labels, transform=None, target_transform=None,
                 loader=opencv_loader, ft_size=(10,10)):
        super().__init__(root, labels, transform, 
                         target_transform, loader)
        self.ft_size = ft_size
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.labels.iloc[idx, 0])
        sample = self.loader(path)
        target = self.labels.iloc[idx, 1]
        
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None --> ', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, self.ft_size)
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')
    

def get_train_valid(conf):
    
    train_transform = T.Compose([
        T.ToPILImage(),
        SquarePad(),
        #T.Resize(size = conf.input_size),
        T.RandomResizedCrop(size=tuple(2*[conf.input_size]), 
                            scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomRotation(10),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    
    valid_transform = T.Compose([
        T.ToPILImage(),
        SquarePad(),
        T.Resize(size = conf.input_size),
        T.ToTensor()
    ])
    
    if conf.spoof_categories is not None:
        cat = conf.spoof_categories
        if cat == 'binary':
            target_transform = lambda t: 0 if t == 0 else 1
        else:
            target_transform = lambda t: next(i for i, l in enumerate(cat) if t in l)
    else:
        target_transform = None
    
    root_path = conf.train_path
    train_labels = pd.read_csv(conf.labels_path)
    train_labels, valid_labels = train_test_split(train_labels, test_size=0.2, 
                                                  random_state=20220826)
    train_labels = train_labels.reset_index(drop=True)
    valid_labels = valid_labels.reset_index(drop=True)
    
    train_loader = DataLoader(
        CelebADatasetFT(root_path, train_labels,
                        train_transform, target_transform, 
                        ft_size=conf.ft_size), 
        batch_size=conf.batch_size,
        shuffle=True, pin_memory=True, #num_workers=8
    )
    valid_loader = DataLoader(
        CelebADataset(root_path, valid_labels,
                      valid_transform, target_transform), 
        batch_size=conf.batch_size,
        shuffle=True, pin_memory=True, #num_workers=8
    )
    
    return train_loader, valid_loader
