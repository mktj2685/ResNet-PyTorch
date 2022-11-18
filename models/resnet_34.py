import torch
import torch.nn as nn
from models.residual_block import ResidualBlock


class ResNet34(nn.Module):

    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # conv1 output size : 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),                  # conv2 output size : 56 x 56
            ResidualBlock(64, 64, 1, False),        # conv2_1
            ResidualBlock(64, 64, 1, False),        # conv2_2
            ResidualBlock(64, 64, 1, False),        # conv2_3
            ResidualBlock(64, 128, 2, False),       # conv3_1 output size : 28 x 28  (downsample)
            ResidualBlock(128, 128, 1, False),      # conv3_2
            ResidualBlock(128, 128, 1, False),      # conv3_3
            ResidualBlock(128, 128, 1, False),      # conv3_4
            ResidualBlock(128, 256, 2, False),      # conv4_1 output size : 14 x 14  (downsample)
            ResidualBlock(256, 256, 1, False),      # conv4_2
            ResidualBlock(256, 256, 1, False),      # conv4_3
            ResidualBlock(256, 256, 1, False),      # conv4_4
            ResidualBlock(256, 256, 1, False),      # conv4_5
            ResidualBlock(256, 256, 1, False),      # conv4_6
            ResidualBlock(256, 512, 2, False),      # conv5_1 output size : 7 x 7  (downsample)
            ResidualBlock(512, 512, 1, False),      # conv5_2
            ResidualBlock(512, 512, 1, False),      # conv5_3
            nn.AvgPool2d(7)                         # output size : 1 x 1
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    x = torch.randn(1, 3, 224, 224).to(device)
    model = ResNet34(1000).to(device)
    out = model(x)
    print(out.shape)
