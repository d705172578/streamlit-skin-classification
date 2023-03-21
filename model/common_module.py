from torch import nn
import torch.nn.functional as F


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.res = True if in_channels == out_channels else False

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out + x if self.res else out


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpSampling, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x,):
        x = self.model(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel=2048, hidden=512):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y
