import pdb
import re
from torch.autograd import Variable
import modelCAVE_stage2_noSpaG as model
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
    parser.add_argument('--input_dir', help='test lrms image name', default='dataset/dataset-CAVE/')
    parser.add_argument('--mspath', help='test lrms image name', default='dataset/dataset-CAVE/test/HRMS/')
    parser.add_argument('--hspath', help='test hrpan image name', default='dataset/dataset-CAVE/test/LRHS/')
    parser.add_argument('--modelpath', help='output model dir', default='/mnt/2040bb09-e64f-4d46-af33-3db5783aadf1/yj/paper/compare_three/MGF-Net_unfold/output/model-CAVE_stage2_noSpaG/epoch_2000.pth')
    parser.add_argument('--saveimgpath', help='output model dir', default='result/CAVE_stage2_noSpaG/2000/')
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--testBatchSize', default=4)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    opt = parser.parse_args()
    test_start_time = time.time()
    test_set = dataloader.get_test_set(opt.input_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)
    net = model.Net().to(opt.device)
    # input1 = (torch.randn(1, 31, 64, 64).to(opt.device))
    # input2 = (torch.randn(1, 3, 256, 256).to(opt.device))
    # flops, params = profile(net, (input1, input2))
    # print('flops:', flops, 'params:', params)
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




# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mspath', help='test lrms image name', default='dataset/dataset-CAVE/test/HRMS/')
#     parser.add_argument('--hspath', help='test hrpan image name', default='dataset/dataset-CAVE/test/LRHS/')
#     parser.add_argument('--modelpath', help='output model dir', default='/mnt/bb6fb707-6986-48d7-801b-4fe78eefd232/yj/Double_Couple/D_C/output/model-CAVE/best.pth')
#     parser.add_argument('--saveimgpath', help='output model dir', default='result/')
#     parser.add_argument('--device', default=torch.device('cuda:0'))
#     # parser.add_argument('--msi_channels', type=int, default=6)
#     # parser.add_argument('--hsi_channels', type=int, default=191)
#     opt = parser.parse_args()
#
#     net = model.DCNet().to(opt.device)
#     modelname = opt.modelpath
#     net.load_state_dict(torch.load(modelname))
#     S = torch.tensor(sio.loadmat('/mnt/bb6fb707-6986-48d7-801b-4fe78eefd232/yj/Double_Couple/D_C/dataset/dataset-CAVE/S/S_CAVE.mat')['S'].astype(float))
#     S = Variable(S.to(torch.float32)).to(opt.device)
#
#     with torch.no_grad():
#         LRHSTest = sio.loadmat(opt.hspath + 'LRHS' + '_chart_and_stuffed_toy_1.mat')['LRHS']
#         LRHSTest=functions.test_matRead(LRHSTest, opt)
#         HRMSTest = sio.loadmat(opt.mspath + 'HRMS' + '_chart_and_stuffed_toy_1.mat')['HRMS']
#         HRMSTest = functions.test_matRead(HRMSTest, opt)
#         in_s = net(LRHSTest, HRMSTest, S)
#         output = functions.convert_image_np((in_s.detach()), opt).astype(np.uint16)
#         sio.savemat('/mnt/bb6fb707-6986-48d7-801b-4fe78eefd232/yj/Double_Couple/D_C/output/result/image/hrhsbest.mat', {'middle_image': output})
#
# if __name__ == '__main__':
#     main()