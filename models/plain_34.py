import torch
import torch.nn as nn
from models.plain_block import PlainBlock


class Plain34(nn.Module):

    def __init__(self, num_classes):
        super(Plain34, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),              # output size : 112 x 112
            nn.MaxPool2d(3, 2, 1),                  # output size : 56 x 56
            PlainBlock(64, 64, 64, 1, True),        # conv2_1
            PlainBlock(64, 64, 64, 1, True),        # conv2_2
            PlainBlock(64, 64, 64, 1, True),        # conv2_3
            PlainBlock(64, 128, 128, 2, True),      # conv3_1   output size : 28 x 28
            PlainBlock(128, 128, 128, 1, True),     # conv3_2
            PlainBlock(128, 128, 128, 1, True),     # conv3_3
            PlainBlock(128, 128, 128, 1, True),     # conv3_4
            PlainBlock(128, 256, 256, 2, True),     # conv4_1   output size : 14 x 14
            PlainBlock(256, 256, 256, 1, True),     # conv4_2
            PlainBlock(256, 256, 256, 1, True),     # conv4_3
            PlainBlock(256, 256, 256, 1, True),     # conv4_4
            PlainBlock(256, 256, 256, 1, True),     # conv4_5
            PlainBlock(256, 256, 256, 1, True),     # conv4_6
            PlainBlock(256, 512, 512, 2, True),     # conv5_1   output size : 7 x 7
            PlainBlock(512, 512, 512, 1, True),     # conv5_2
            PlainBlock(512, 512, 512, 1, True),     # conv5_3
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
    model = Plain34(1000).to(device)
    out = model(x)
    print(out.shape)
