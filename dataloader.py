import pdb
from os.path import join
import os
import torch
from torch.utils.data import Dataset
from skimage import io as skimage
import numpy as np
import re
import scipy.io as sio

def normalize(data):
    data=data.astype(np.float32)
    for i in range(len(data)):
        data[i, :, :] -= data[i, :, :].min()
        data[i, :, :] /= data[i, :, :].max()
    return data

def matRead(data):
    data=data.transpose(2, 0, 1)
    # data = data/32701.
    data = data / 64000.
    # data = data / 0.07
    # data=normalize(data)
    data=torch.from_numpy(data)
    # data=(data-0.5)*2
    # data=data.clamp(-1,1)

    return data

class Dataset(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.LRHSpath = path + 'LRHS/'
        self.HRMSpath = path + 'HRMS/'
        self.HRHSpath = path + 'HRHS/'

        files = os.listdir(self.LRHSpath)
        img_list = []
        for index in files:
            num = re.split(r'[S.]', index)[1]
            img_list.append(num)
        self.img_list = img_list

    def __getitem__(self, index):

        fn = self.img_list[index]
        LRHSBatch = sio.loadmat(self.LRHSpath+'LRHS' + str(fn) + '.mat')['lrhs']
        # LRHSBatch = sio.loadmat(self.LRHSpath + 'LRHS' + str(fn) + '.mat')['LRHS']
        LRHSBatch = matRead(LRHSBatch)

        HRMSBatch = sio.loadmat(self.HRMSpath+'HRMS' + str(fn) + '.mat')['hrms']
        # HRMSBatch = sio.loadmat(self.HRMSpath + 'HRMS' + str(fn) + '.mat')['HRMS']
        HRMSBatch = matRead(HRMSBatch)

        HRHSBatch = sio.loadmat(self.HRHSpath+'HRHS' + str(fn) + '.mat')['hrhs']
        # HRHSBatch = sio.loadmat(self.HRHSpath + 'HRHS' + str(fn) + '.mat')['HRHS']
        HRHSBatch = matRead(HRHSBatch)

        return LRHSBatch, HRMSBatch, HRHSBatch

    def __len__(self):
        return len(self.img_list)

class Datasetval(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.LRHSpath = path + 'LRHS/'
        self.HRMSpath = path + 'HRMS/'
        self.HRHSpath = path + 'HRHS/'

        files = os.listdir(self.LRHSpath)
        img_list = []
        for index in files:
            num = re.split(r'[S.]', index)[1]
            img_list.append(num)
        self.img_list = img_list

    def __getitem__(self, index):
        fn = self.img_list[index]
        # LRHSBatch = sio.loadmat(self.LRHSpath + 'LRHS' + str(fn) + '.mat')['lrhs']
        LRHSBatch = sio.loadmat(self.LRHSpath + 'LRHS' + str(fn) + '.mat')['LRHS']
        LRHSBatch = matRead(LRHSBatch)

        # HRMSBatch = sio.loadmat(self.HRMSpath + 'HRMS' + str(fn) + '.mat')['hrms']
        HRMSBatch = sio.loadmat(self.HRMSpath + 'HRMS' + str(fn) + '.mat')['HRMS']
        HRMSBatch = matRead(HRMSBatch)

        # HRHSBatch = sio.loadmat(self.HRHSpath + 'HRHS' + str(fn) + '.mat')['hrhs']
        HRHSBatch = sio.loadmat(self.HRHSpath + 'HRHS' + str(fn) + '.mat')['HRHS']
        HRHSBatch = matRead(HRHSBatch)

        return LRHSBatch, HRMSBatch, HRHSBatch, fn

    def __len__(self):
        return len(self.img_list)

def get_training_set(root_dir):
    train_dir = join(root_dir, "train/")
    return Dataset(train_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, "val/")
    return Datasetval(val_dir)

def get_test_set(root_dir):
    test_dir = join(root_dir, "test/")
    return Datasetval(test_dir)