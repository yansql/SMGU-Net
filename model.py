import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import scipy.io as sio
from torch.autograd import Variable

class SpeUB(nn.Module):
    def __init__(self, hsi_channels, msi_channels):
        super(SpeUB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=msi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hsi_channels, out_channels=msi_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=msi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, HRMS):
        Y1 = self.conv1(HRMS)
        Y2 = self.conv2(Y1)
        mid = HRMS - Y2
        Y3 = self.conv3(mid)
        out = Y1 + Y3
        return out

class SpaUB(nn.Module):
    def __init__(self, hsi_channels):
        super(SpaUB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, LRHS):
        X1 = self.deconv1(LRHS)
        X2 = self.conv1(X1)
        mid = LRHS - X2
        X3 = self.deconv2(mid)
        out = X1 + X3
        return out

class Prior(nn.Module):
    def __init__(self, hsi_channels):
        super(Prior, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_channels=hsi_channels*3, out_channels=hsi_channels, kernel_size=1, stride=1, padding=0),
        )
    def forward(self, H):
        out1 = self.conv1(H)
        out2 = self.conv2(H)
        out3 = self.conv3(H)
        out_cat = torch.cat([out1, out2, out3], dim=1)
        out = self.cat_conv(out_cat)
        result = H + out
        return result
class SpaEncoder(nn.Module):
    def __init__(self, hsi_channels):
        super(SpaEncoder, self).__init__()
        self.E1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
                                )
        self.E2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
                                )
    def forward(self, x):
        Ea1 = x
        Ea2 = self.E1(Ea1)
        Ea3 = self.E2(Ea2)
        return Ea1, Ea2, Ea3

class SpaDecoder(nn.Module):
    def __init__(self, hsi_channels):
        super(SpaDecoder, self).__init__()
        self.D1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
                                )
        self.D2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64+hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
                                )
    def forward(self, Ea1, Ea2, Ea3):
        D1 = self.D1(torch.cat([Ea3, Ea2], dim=1))
        D2 = self.D2(torch.cat([Ea1, D1], dim=1))
        return D2

class SpeEncoder(nn.Module):
    def __init__(self):
        super(SpeEncoder, self).__init__()
    def forward(self, x):
        Ee1 = x
        Ee2 = F.avg_pool2d(Ee1, kernel_size=2, stride=2)
        Ee3 = F.avg_pool2d(Ee2, kernel_size=2, stride=2)
        return Ee1, Ee2, Ee3

class SpeDecoder(nn.Module):
    def __init__(self, hsi_channels):
        super(SpeDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels*3, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, Ee1, Ee2, Ee3):
        D1 = torch.cat([Ee2, F.interpolate(Ee3, scale_factor=2, mode='bicubic', align_corners=True)], dim=1)
        D2 = torch.cat([Ee1, F.interpolate(D1, scale_factor=2, mode='bicubic', align_corners=True)], dim=1)
        out = self.conv(D2)
        return out

class RDB(nn.Module):
    def __init__(self, hsi_channels):
        super(RDB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=hsi_channels+hsi_channels, out_channels=hsi_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels*2, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels*3, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels*4, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels*5, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
        )
    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 + x

class Spa_enhance(nn.Module):
    def __init__(self, hsi_channels):
        super(Spa_enhance, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels+hsi_channels+hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
        )
    def forward(self, out1, LRHS_UP):
        X1 = self.conv1(LRHS_UP)
        Y1 = self.conv2(out1)
        XY = X1 * Y1
        cat = torch.cat([X1, XY, Y1], dim=1)
        out = self.conv3(cat)
        return out

class Spe_enhance(nn.Module):
    def __init__(self, hsi_channels):
        super(Spe_enhance, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels+hsi_channels+hsi_channels, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
        )
    def forward(self, out2, HRMS_UP):
        X1 = self.conv1(out2)
        Y1 = self.conv2(HRMS_UP)
        XY = X1 * Y1
        cat = torch.cat([X1, XY, Y1], dim=1)
        out = self.conv3(cat)
        return out

class SpaIGM(nn.Module):
    def __init__(self, hsi_channels):
        super(SpaIGM, self).__init__()
        self.SpaEncoder = SpaEncoder(hsi_channels)
        self.SpaDecoder = SpaDecoder(hsi_channels)
        self.Spa_enhance = Spa_enhance(hsi_channels)
        self.RDB = RDB(hsi_channels)
    def forward(self, LRHS_UP, HRMS_UP):
        LM_cat1 = torch.cat((LRHS_UP, HRMS_UP), dim=1)
        Ea1, Ea2, Ea3 = self.SpaEncoder(HRMS_UP)
        out1 = self.SpaDecoder(Ea1, Ea2, Ea3)
        Spa_enhancement = self.Spa_enhance(out1, LRHS_UP)
        LM_RDB1 = self.RDB(LM_cat1)
        result1 = Spa_enhancement + LM_RDB1
        return result1

class SpeIGM(nn.Module):
    def __init__(self, hsi_channels):
        super(SpeIGM, self).__init__()
        self.SpeEncoder = SpeEncoder()
        self.SpeDecoder = SpeDecoder(hsi_channels)
        self.Spe_enhance = Spe_enhance(hsi_channels)
        self.RDB = RDB(hsi_channels)
    def forward(self, LRHS_UP, HRMS_UP):
        LM_cat2 = torch.cat((LRHS_UP, HRMS_UP), dim=1)
        Ee1, Ee2, Ee3 = self.SpeEncoder(LRHS_UP)
        out2 = self.SpeDecoder(Ee1, Ee2, Ee3)
        Spe_enhancement = self.Spe_enhance(out2, HRMS_UP)
        LM_RDB2 = self.RDB(LM_cat2)
        result2 = Spe_enhancement + LM_RDB2
        return result2

class Fidelity1(nn.Module):
    def __init__(self):
        super(Fidelity1, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor([1]),requires_grad=True)
        self.mu = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
    def forward(self, Hk, L, S1, A):
        out = Hk - self.mu*((1+self.alpha)*Hk - L - S1 - self.alpha*A)
        return out

class Fidelity2(nn.Module):
    def __init__(self):
        super(Fidelity2, self).__init__()
        self.beta = torch.nn.Parameter(torch.Tensor([1]),requires_grad=True)
        self.ro = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
    def forward(self, Hk, M, S2, A):
        out = Hk - self.ro*((1+self.beta)*Hk - M - S2 - self.beta*A)
        return out


class Fusion(nn.Module):
    def __init__(self, hsi_channels):
        super(Fusion, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels*2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels*2, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels * 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels * 2, out_channels=hsi_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, out1, out2):
        out11 = self.conv2(self.conv1(out1))
        out22 = self.conv4(self.conv3(out2))
        out12 = out1*out22
        out21 = out2*out11
        out1_fin = out12 + out1
        out2_fin = out21 + out2
        fin = out1_fin + out2_fin
        return fin

class Net(nn.Module):
    def __init__(self, niter=2, hsi_channels=31, msi_channels=3):
        super(Net, self).__init__()
        self.niter = niter
        self.SpeUB = SpeUB(hsi_channels, msi_channels)
        self.SpaUB = SpaUB(hsi_channels)
        self.Prior = Prior(hsi_channels)
        self.SpaIGM = SpaIGM(hsi_channels)
        self.SpeIGM = SpeIGM(hsi_channels)
        self.Fusion = Fusion(hsi_channels)
        self.Fidelity1 = Fidelity1()
        self.Fidelity2 = Fidelity2()
    def forward(self, LRHS, HRMS):
        LRHS_UP = self.SpaUB(LRHS)
        HRMS_UP = self.SpeUB(HRMS)
        LRHS_INUP = F.interpolate(LRHS, scale_factor=4, mode='bicubic', align_corners=True)
        for i in range(self.niter):
            Prior_in = self.Prior(LRHS_INUP)
            result1 = self.SpaIGM(LRHS_UP, HRMS_UP)
            result2 = self.SpeIGM(LRHS_UP, HRMS_UP)
            F1 = self.Fidelity1(LRHS_INUP, LRHS_UP, result1, Prior_in)
            F2 = self.Fidelity2(LRHS_INUP, HRMS_UP, result2, Prior_in)
            out = self.Fusion(F1, F2)
            LRHS_INUP = out
        return out