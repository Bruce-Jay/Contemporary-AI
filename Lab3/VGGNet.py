import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, conv_num, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.net = nn.Sequential()
        for i in range(conv_num):
            self.net.add_module(
                "conv_{0}".format(i), nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )
            in_channels = out_channels
        self.net.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.net(x)


class VGGNet11(nn.Module):
    def __init__(self, dropout):
        super(VGGNet11, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            VGGBlock(1, 1, 64),
            VGGBlock(1, 64, 128),
            VGGBlock(2, 128, 256),
            VGGBlock(2, 256, 512),
            VGGBlock(2, 512, 512),
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=10)
        )

    def forward(self, x):
        return self.net(x)


class VGGNet19(nn.Module):
    def __init__(self, dropout):
        super(VGGNet19, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            VGGBlock(2, 1, 64),
            VGGBlock(2, 64, 128),
            VGGBlock(4, 128, 256),
            VGGBlock(4, 256, 512),
            VGGBlock(4, 512, 512),
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=10)
        )

    def forward(self, x):
        return self.net(x)