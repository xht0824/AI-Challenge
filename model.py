import torchvision 
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.optim
from torchsummary import summary
import torch.nn.functional as F

class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(5, 512)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(512, 500)  # 2个隐层
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(500, 490)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(490, 480)
        self.tanh4 = torch.nn.Tanh()
        self.linear5 = torch.nn.Linear(480, 470)
        self.tanh5 = torch.nn.Tanh()
        self.linear6 = torch.nn.Linear(470, 460)
        self.tanh6 = torch.nn.Tanh()
        self.linear7 = torch.nn.Linear(460, 450)
        self.tanh7 = torch.nn.Tanh()
        self.linear8 = torch.nn.Linear(450, 1)
        self.tanh8 = torch.nn.Tanh()

    def forward(self,x):
        out=self.linear1(x)
        out=self.tanh1(out)
        out=self.linear2(out)
        out=self.tanh2(out)
        out=self.linear3(out)
        out=self.tanh3(out)
        out=self.linear4(out)
        out=self.tanh4(out)
        out=self.linear5(out)
        out=self.tanh5(out)
        out=self.linear6(out)
        out=self.tanh6(out)
        out=self.linear7(out)
        out=self.tanh7(out)
        out=self.linear8(out)
        out=self.tanh8(out)
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, size=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        if size is not None:
            out_ft = torch.zeros(batchsize, self.out_channels, size[0], size[1] // 2 + 1, dtype=torch.cfloat,
                                 device=x.device)
        else:
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                                 device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if size is not None:
            x = torch.fft.irfft2
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FNORecon(nn.Module):
    def __init__(self, sensor_num, fc_size=(12, 12), out_size=(100, 100), modes1=24, modes2=24, width=32):
        super(FNORecon, self).__init__()
        self.fc_size = fc_size
        self.out_size = out_size
        self.fc = nn.Sequential(
            nn.Linear(sensor_num, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 32 * 32),
        )
        self.conv_smooth = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
        )
        self.embedding = nn.Conv2d(1, 32, kernel_size=1)
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Sequential(
            nn.Linear(10000, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        N, _, _ = x.shape
        x = self.fc(x).reshape(N, 1, 32, 32)
        x = self.embedding(x)
        x = self.conv_smooth(F.interpolate(x, scale_factor=2))

        x = F.interpolate(x, self.out_size)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(N, 1, 10000)
        x = self.fc3(x)

        return x
    
# net = FNORecon(sensor_num=5).cuda()
# # net = MLP().cuda()
# summary(net,(1, 5))