import torch.nn as nn
import torch.nn.functional as F
import torch
from .common_module import SELayer


class UnfoldPooling(nn.Module):
    def __init__(self, ratio=2):
        super(UnfoldPooling, self).__init__()
        self.ratio = ratio
        self.unfold = nn.Unfold(kernel_size=self.ratio, stride=self.ratio)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.unfold(x)
        x = x.view(x.size(0), x.size(1), h // self.ratio, w // self.ratio)
        return x


class FoldSampling(nn.Module):
    def __init__(self, out_size, ratio=2):
        super(FoldSampling, self).__init__()
        self.ratio = ratio
        self.fold = nn.Fold(output_size=(out_size, out_size), kernel_size=self.ratio, stride=self.ratio)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        return self.fold(x)


class ResDoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        return out1 + out2

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class SEDoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEDoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.cbam = SELayer(channel=out_channels, hidden=out_channels//2)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.cbam(out)

        return out


class CBAMConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAMConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.cbam = CBAM(channel=out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.cbam(F.relu(self.bn1(self.conv1(x))))


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(InceptionBlock, self).__init__()
        if out_channel % 32 != 0:
            raise ValueError('the value %d of out_channel is invalid' % out_channel)
        base = out_channel // 32
        branch1x1out = base * 5
        branch3x3out = base * 6
        middleout = base * 7
        poolout = base * 3

        self.branch1x1 = BasicConv2d(in_channels, branch1x1out, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, branch3x3out, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(branch3x3out, branch3x3out, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(branch3x3out, branch3x3out, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, middleout, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(middleout, branch3x3out, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(branch3x3out, branch3x3out, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(branch3x3out, branch3x3out, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, poolout, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
