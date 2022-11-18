import torch
import torch.nn as nn
from models.residual_block import ResidualBlock


class ResNet50(nn.Module):

    def __init__(self, num_classes:int):
        super(ResNet50, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # output size : 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),                  # output size : 56 x 56
            ResidualBlock(64, 256, 1, True),        # conv2_1
            ResidualBlock(256, 256, 1, True),       # conv2_2
            ResidualBlock(256, 256, 1, True),       # conv2_3
            ResidualBlock(256, 512, 2, True),       # conv3_1 (downsample)
            ResidualBlock(512, 512, 1, True),       # conv3_2
            ResidualBlock(512, 512, 1, True),       # conv3_3
            ResidualBlock(512, 512, 1, True),       # conv3_4
            ResidualBlock(512, 1024, 2, True),      # conv4_1 (downsample)
            ResidualBlock(1024, 1024, 1, True),     # conv4_2
            ResidualBlock(1024, 1024, 1, True),     # conv4_3
            ResidualBlock(1024, 1024, 1, True),     # conv4_4
            ResidualBlock(1024, 1024, 1, True),     # conv4_5
            ResidualBlock(1024, 1024, 1, True),     # conv4_6
            ResidualBlock(1024, 2048, 2, True),     # conv5_1 (downsample)
            ResidualBlock(2048, 2048, 1, True),     # conv5_2
            ResidualBlock(2048, 2048, 1, True),     # conv5_3
            nn.AvgPool2d(7)                         # output size : 1 x 1
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    x = torch.randn(1, 3, 224, 224).to(device)
    model = ResNet50(1000).to(device)
    out = model(x)
    print(out.shape)
