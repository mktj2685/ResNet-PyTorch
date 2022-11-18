import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    def __init__(self, in_ch:int, conv_ch:int, out_ch:int, stride:int, activate:bool) -> None:
        super(Bottleneck, self).__init__()
        self.activate = activate
        self.conv1 = nn.Conv2d(in_ch, conv_ch, 1, 1)
        self.bn1 = nn.BatchNorm2d(conv_ch)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(conv_ch, conv_ch, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(conv_ch)
        self.conv3 = nn.Conv2d(conv_ch, out_ch, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.activate:
            x = self.relu(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 256, 112, 112)
    block = Bottleneck(256, 64, 256, 2, True)
    x = block(x)
    print(x.shape)