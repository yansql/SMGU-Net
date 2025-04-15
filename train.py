import argparse
import pdb
import scipy.io as sio
import modelCAVE_stage2_no_Up as model
import torch
import torch.nn as nn
import functions
import time
import os
import copy
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='dataset/dataset-CAVE')
    parser.add_argument('--val_dir', help='testing_data', default='dataset/dataset-CAVE')
    parser.add_argument('--outputs_dir', help='output model dir', default='output/model-CAVE_noUp')
    parser.add_argument('--batchSize', default=1)
    parser.add_argument('--testBatchSize', default=1)
    parser.add_argument('--epoch', default=2001)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=1)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr', type=float, default=0.0001, help='G‘s learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='scheduler gamma')
    # parser.add_argument('--msi_channels', type=int, default=3)
    # parser.add_argument('--hsi_channels', type=int, default=31)
    opt = parser.parse_args()
    train_start_time = time.time()
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    train_set = dataloader.get_training_set(opt.input_dir)
    val_set = dataloader.get_val_set(opt.val_dir)


    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    # 网络初始化：
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Net = model.Net().to(opt.device)
    # Net = torch.nn.DataParallel(model.Net(opt).cuda())
    for module in Net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    # 建立优化器
    optimizer = torch.optim.Adam(Net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=opt.gamma)


    loss = torch.nn.L1Loss().to(opt.device)

    best_weights = copy.deepcopy(Net.state_dict())
    best_epoch = 0
    best_SAM = 1.0
    trainvalid_txt = '/mnt/2040bb09-e64f-4d46-af33-3db5783aadf1/yj/paper/compare_three/MGF-Net_unfold/valid_stage2_noUp'
    for i in range(opt.epoch):
        # train
        Net.train()
        epoch_losses = functions.AverageMeter()
        batch_time = functions.AverageMeter()
        end = time.time()
        for batch_idx, (hsBatch, msBatch, hrhsBatch) in enumerate(train_loader):
            if torch.cuda.is_available():
                hsBatch, msBatch, hrhsBatch = hsBatch.to(opt.device), msBatch.to(opt.device), hrhsBatch.to(opt.device)
                hsBatch = Variable(hsBatch.to(torch.float32))
                msBatch = Variable(msBatch.to(torch.float32))
                hrhsBatch = Variable(hrhsBatch.to(torch.float32))

            N = len(train_loader)
            Net.zero_grad()

            result = Net(hsBatch, msBatch)
            outLoss = loss(result, hrhsBatch)
            outLoss.backward(retain_graph=True)
            optimizer.step()
            epoch_losses.update(outLoss.item(), hrhsBatch.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_idx + 1) % 100 == 0:
                training_state = '  '.join(
                    ['Epoch: {}', '[{} / {}]', 'Loss: {:.6f}']
                )
                training_state = training_state.format(
                    i, batch_idx, N, outLoss
                )
                print(training_state)

        print('%d epoch: loss is %.6f, epoch time is %.4f' % (i, epoch_losses.avg, batch_time.avg))
        if i % 50 == 0:
            torch.save(Net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(i)))

        Net.eval()
        epoch_SAM = functions.AverageMeter()
        with torch.no_grad():
            for j, (hsTest, msTest, hrhsTest, fn) in enumerate(val_loader):
                if torch.cuda.is_available():
                    hsTest, msTest, hrhsTest = hsTest.to(opt.device), msTest.to(opt.device), hrhsTest.to(opt.device)
                    hsTest = Variable(hsTest.to(torch.float32))
                    msTest = Variable(msTest.to(torch.float32))
                    hrhsTest = Variable(hrhsTest.to(torch.float32))

                mp = Net(hsTest, msTest)
                # test_SAM = 0.5*loss1(mp1, hrhsTest) + 0.5*loss2(mp2, hrhsTest) + loss(mp, hrhsTest)
                test_SAM = functions.SAM(mp, hrhsTest)
                epoch_SAM.update(test_SAM, hsTest.shape[0])
            print('eval SAM: {:.6f}'.format(epoch_SAM.avg))
        if epoch_SAM.avg < best_SAM:
            best_epoch = i
            best_SAM = epoch_SAM.avg
            best_weights = copy.deepcopy(Net.state_dict())
            torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))
        print('best epoch:{:.0f}'.format(best_epoch))
        outputt = "step: %d, outLoss: %f, eval SAM: %f, best_epoch: %d" % (i, epoch_losses.avg, epoch_SAM.avg, best_epoch)
        with open(trainvalid_txt, "a+") as f:
            f.write(outputt + '\n')
            f.close
        scheduler.step()
        train_end_time = (time.time() - train_start_time) / 3600
        print(f'train all time: {train_end_time :.4f} hour')
    print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    # torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))matRead