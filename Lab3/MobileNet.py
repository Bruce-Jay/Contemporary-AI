import torch.nn as nn

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            MobileBlock(in_channels=32, out_channels=64),
            MobileBlock(in_channels=64, out_channels=128, stride=2),
            MobileBlock(in_channels=128, out_channels=128),
            MobileBlock(in_channels=128, out_channels=256, stride=2),
            MobileBlock(in_channels=256, out_channels=256),
            MobileBlock(in_channels=256, out_channels=512, stride=2),
            MobileBlock(in_channels=512, out_channels=512),
            MobileBlock(in_channels=512, out_channels=512),
            MobileBlock(in_channels=512, out_channels=512),
            MobileBlock(in_channels=512, out_channels=512),
            MobileBlock(in_channels=512, out_channels=512),
            MobileBlock(in_channels=512, out_channels=1024, stride=2),
            MobileBlock(in_channels=1024, out_channels=1024),
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        return self.net(x)