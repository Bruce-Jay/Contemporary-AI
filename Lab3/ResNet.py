import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, res_conv=False):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        if res_conv:
            self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.res_conv = None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        if self.res_conv:
            x = self.res_conv(x)
        y = y + x
        return self.relu(y)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.net.add_module('res_1', nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, stride=1),
            ResBlock(in_channels=64, out_channels=64, stride=1)
        ))
        self.net.add_module('res_2', nn.Sequential(
            ResBlock(in_channels=64, out_channels=128, stride=2, res_conv=True),
            ResBlock(in_channels=128, out_channels=128, stride=1)
        ))
        self.net.add_module('res_3', nn.Sequential(
            ResBlock(in_channels=128, out_channels=256, stride=2, res_conv=True),
            ResBlock(in_channels=256, out_channels=256, stride=1)
        ))
        self.net.add_module('res_4', nn.Sequential(
            ResBlock(in_channels=256, out_channels=512, stride=2, res_conv=True),
            ResBlock(in_channels=512, out_channels=512, stride=1)
        ))
        self.net.add_module('output', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10)
        ))

    def forward(self, x):
        return self.net(x)