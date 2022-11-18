import torch
import torch.nn as nn
from models.residual_block import ResidualBlock


class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),                  # output size : 112 x 112
            nn.MaxPool2d(3, 2, 1),                      # output size : 56 x 56
            ResidualBlock(64, 64, 64, 1, False),
            ResidualBlock(64, 64, 128, 2, False),       # output size : 28 x 28
            ResidualBlock(128, 128, 128, 1, False),
            ResidualBlock(128, 128, 256, 2, False),     # output size : 14 x 14
            ResidualBlock(256, 256, 256, 1, False),
            ResidualBlock(256, 256, 512, 2, False),     # output size : 7 x 7
            ResidualBlock(512, 512, 512, 1, False),
            ResidualBlock(512, 512, 512, 1, False),
            nn.AvgPool2d(7)                             # output size : 1 x 1
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
    model = ResNet18(1000).to(device)
    out = model(x)
    print(out.shape)
