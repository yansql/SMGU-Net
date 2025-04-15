import pdb
import re
from torch.autograd import Variable
import modelCAVE as model
import torch
import functions
import numpy as np
import os
from skimage import io
import argparse
import scipy.io as sio
from thop import profile
import time
import dataloader
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='test lrms image name', default='')
    parser.add_argument('--mspath', help='test lrms image name', default='')
    parser.add_argument('--hspath', help='test hrpan image name', default='')
    parser.add_argument('--modelpath', help='output model dir', default='')
    parser.add_argument('--saveimgpath', help='output model dir', default='')
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--testBatchSize', default=4)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    opt = parser.parse_args()
    test_start_time = time.time()
    test_set = dataloader.get_test_set(opt.input_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)
    net = model.Net().to(opt.device)
    modelname = opt.modelpath
    net.load_state_dict(torch.load(modelname))

    with torch.no_grad():
        for j, (LRHStest, HRMStest, HRHStest, name) in enumerate(test_loader):
            LRHStest, HRMStest, HRHStest = LRHStest.cuda(), HRMStest.cuda(), HRHStest.cuda()
            LRHStest = Variable(LRHStest.to(torch.float32))
            HRMStest = Variable(HRMStest.to(torch.float32))
            HRHStest = Variable(HRHStest.to(torch.float32))
            in_s = net(LRHStest, HRMStest)
            outname = opt.saveimgpath + 'HRHS-' + name[0] + '.mat'
            sio.savemat(outname, {'out': functions.convert_image_np((in_s.detach()), opt)})
        train_end_time = (time.time() - test_start_time)
        print(f'test all time: {train_end_time :.4f} s')


if __name__ == '__main__':
    main()

