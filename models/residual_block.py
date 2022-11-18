import torch
import torch.nn as nn


class Plain(nn.Module):

    def __init__(self, in_ch:int, out_ch:int, stride:int) -> None:
        super(Plain, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class Bottleneck(nn.Module):

    def __init__(self, in_ch:int, out_ch:int, stride:int, scale:int=4) -> None:
        super(Bottleneck, self).__init__()
        hid_ch = out_ch // scale
        self.conv1 = nn.Conv2d(in_ch, hid_ch, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hid_ch)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hid_ch)
        self.conv3 = nn.Conv2d(hid_ch, out_ch, 1, 1, bias=False)
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
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride:int, bottleneck:bool):
        super(ResidualBlock, self).__init__()
        self.layers = Bottleneck(in_ch, out_ch, stride) if bottleneck else Plain(in_ch, out_ch, stride)
        # see https://stackoverflow.com/questions/55688645/how-downsample-work-in-resnet-in-pytorch-code
        self.proj = None
        if in_ch != out_ch or stride > 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
                nn.BatchNorm2d(out_ch)
            )    
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.layers(x)
        if self.proj:
            x = self.proj(x)
        out = out + x
        out = self.relu(out)
        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    x = torch.randn(3, 64, 56, 56).to(device)
    # model = ResidualBlock(64, 128, False).to(device)
    model = ResidualBlock(64, 128, True).to(device)
    out = model(x)
    print(out.shape)
