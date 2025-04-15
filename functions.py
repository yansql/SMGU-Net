import skimage.io as skimage
import torch
import numpy as np
from numpy import *
import pdb
from sewar.full_ref import sam
import torch.nn.functional as F

def test_matRead(data,opt):
    data = data[None, :, :, :]
    # data = data.transpose(0, 3, 1, 2)/32701.#WSDC
    data = data.transpose(0, 3, 1, 2) / 64000. #CAVE
    # data = data.transpose(0, 3, 1, 2) / 0.07
    # data = data.transpose(0, 3, 1, 2)/8000. #PAVIA
    data = torch.from_numpy(data)
    data = data.type(torch.cuda.FloatTensor)
    data = data.to(opt.device)
    return data

def getBatch(hsBatch, msBatch, hrhsBatch, bs):
    N = hrhsBatch.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    hrmsBatch = msBatch[batchIndex, :, :, :]
    gtBatch = hrhsBatch[batchIndex, :, :, :]
    lrhsBatch = hsBatch[batchIndex, :, :, :]
    return lrhsBatch, hrmsBatch, gtBatch

def getTest(hrms, label, gt_data, lrhs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=1)
    hrmsBatch = torch.linalg.invhrms[batchIndex, :, :, :]
    labelBatch = label[batchIndex, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    lrhsBatch = lrhs[batchIndex, :, :, :]
    return hrmsBatch, labelBatch, gtBatch, lrhsBatch

def convert_image_np(inp,opt):
    inp = inp[-1, :, :, :]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = np.clip(inp/ 2 + 0.5,0,1)
    inp = np.clip(inp, 0, 1)
    inp = (inp) * 64000.
    return inp

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img, hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.numpy()
    sr_img = sr_img[-1, :, :, :]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value

def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data

def calc_psnr(img_tgt,img_fus):
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[0])
    img_fus = img_fus.reshape(-1,img_fus.shape[0])
    mse = torch.mean(torch.square(img_tgt-img_fus))
    img_max = torch.max(img_tgt)
    psnr = 10.0 * torch.log10(img_max**2/mse)
    return psnr
