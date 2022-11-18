import torch
import torch.nn as nn
from models.plain_block import PlainBlock
from models.bottleneck import Bottleneck


class ResidualBlock(nn.Module):

    def __init__(self, in_ch: int, hid_ch:int, out_ch: int, stride:int, bottleneck:bool):
        super(ResidualBlock, self).__init__()
        # NOTE If bottleneck is False, hid_ch is ignored.
        self.layers = Bottleneck(in_ch, hid_ch, out_ch, stride, False) if bottleneck else PlainBlock(in_ch, out_ch, stride, False)
        # see https://stackoverflow.com/questions/55688645/how-downsample-work-in-resnet-in-pytorch-code
        # If stride is larger than 1 or the number of input channels is different from the number of output channels,
        # conv1x1 matches the shape of the input tensor to the shape of the output tensor.
        if stride > 1 or in_ch != out_ch:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.adjust = None
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.layers(x)
        if self.adjust:
            x = self.adjust(x)
        out = out + x
        out = self.relu(out)
        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    x = torch.randn(3, 64, 56, 56).to(device)
    # model = ResidualBlock(64, 64, 128, 2, False).to(device)
    model = ResidualBlock(64, 64, 128, 2, True).to(device)
    out = model(x)
    print(out.shape)
