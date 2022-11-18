import torch
import torch.nn as nn


class PlainBlock(nn.Module):

    def __init__(self, in_ch: int, hid_ch:int, out_ch: int, stride:int, activate:bool):
        super(PlainBlock, self).__init__()
        self.activate = activate
        self.conv1 = nn.Conv2d(in_ch, hid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hid_ch)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(hid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.activate:
            x = self.relu(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    x = torch.randn(1, 256, 56, 56).to(device)
    model = PlainBlock(256, 64, 256, 2, True).to(device)
    out = model(x)
    print(out.shape)
