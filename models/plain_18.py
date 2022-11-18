import torch
import torch.nn as nn
from models.plain_block import PlainBlock


class Plain18(nn.Module):

    def __init__(self, num_classes):
        super(Plain18, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # output size : 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),                  # output size : 56 x 56
            PlainBlock(64, 64),                     # conv2_1
            PlainBlock(64, 64),                     # conv2_2
            PlainBlock(64, 128),                    # conv3_1   output size : 28 x 28
            PlainBlock(128, 128),                   # conv3_2
            PlainBlock(128, 256),                   # conv4_1   output size : 14 x 14
            PlainBlock(256, 256),                   # conv4_2
            PlainBlock(256, 512),                   # conv5_1   output size : 7 x 7
            PlainBlock(512, 512),                   # conv5_2
            nn.AvgPool2d(7),                        # output size : 1 x 1
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
    model = Plain18(1000).to(device)
    out = model(x)
    print(out.shape)
