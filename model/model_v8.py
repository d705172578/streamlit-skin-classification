import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from .common_module import *
from .new_module import *

resnet = torchvision.models.resnext50_32x4d(pretrained=True)

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


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256, dl=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=dl[0], dilation=dl[0])
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=dl[1], dilation=dl[1])
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=dl[2], dilation=dl[2])
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)

        out = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block2,
                                              atrous_block3, atrous_block4], dim=1))
        return out

class SEASPP(nn.Module):
    def __init__(self, channel=64, depth=32, hidden=32, dl=[6, 12, 18]):
        super(SEASPP, self).__init__()
        self.block = nn.Sequential(
            ASPP(in_channel=channel, depth=depth, dl=dl),
            nn.InstanceNorm2d(depth),
            nn.ReLU(inplace=True),
            SELayer(depth, hidden),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.block(x)

# 把model_v2的普通输入增强的IUNet的卷积换成带SE的卷积
class ResIUNet(nn.Module):
    def __init__(self, img_size=224):
        super(ResIUNet, self).__init__()
        self.img_size = img_size
        # self.outresize = transforms.Resize(self.img_size)

        self.resize1 = UnfoldPooling(2)
        self.resize2 = UnfoldPooling(4)
        self.resize3 = UnfoldPooling(8)
        self.resize4 = UnfoldPooling(16)

        self.enhance1 = nn.Sequential(self.resize1, SEASPP(channel=12, depth=64, hidden=32, dl=[6, 12, 18])) # CBAMConv2d(12, 64))
        self.enhance2 = nn.Sequential(self.resize2, SEASPP(channel=48, depth=256, hidden=128, dl=[4, 8, 12])) # CBAMConv2d(48, 256))
        self.enhance3 = nn.Sequential(self.resize3, SEASPP(channel=192, depth=512, hidden=256, dl=[3, 6, 9])) # CBAMConv2d(192, 512))
        self.enhance4 = nn.Sequential(self.resize4, SEASPP(channel=768, depth=1024, hidden=512, dl=[2, 4, 6])) # CBAMConv2d(768, 1024))

        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )   # 64
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encode2 = resnet.layer1    # 256
        self.encode3 = resnet.layer2    # 512
        self.encode4 = resnet.layer3    # 1024
        self.encode5 = resnet.layer4    # 2048

        self.decode4 = InceptionBlock(4608, 1024)
        self.decode3 = InceptionBlock(2304, 512)
        self.decode2 = InceptionBlock(1088, 256)
        self.decode1 = InceptionBlock(384, 64)

        self.out = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

        self.deconv5_4 = UpSampling(3072, 1024)
        self.deconv4_3 = UpSampling(1024, 512)
        self.deconv3_2 = UpSampling(512, 256)
        self.deconv2_1 = UpSampling(256, 64)
        self.extra_deconv = UpSampling(64, 32)

        # self.se1 = SEASPP(channel=320, depth=320, hidden=160, dl=[6, 12, 18])
        # self.se2 = SEASPP(channel=832, depth=832, hidden=416, dl=[4, 8, 12])
        # self.se3 = SEASPP(channel=1792, depth=1792, hidden=896, dl=[3, 6, 9])
        # self.se4 = SEASPP(channel=3584, depth=3584, hidden=1792, dl=[2, 4, 6])
        self.se1 = CBAM(320)
        self.se2 = CBAM(832)
        self.se3 = CBAM(1792)
        self.se4 = CBAM(3584)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        # encode
        enhance1 = self.enhance1(x)
        encoded1 = self.encode1(x)
        pool1 = self.maxpool(encoded1 + enhance1)    # [N, 64, 112, 112]

        enhance2 = self.enhance2(x)
        encoded2 = self.encode2(pool1)
        pool2 = encoded2 + enhance2    # [N, 256, 56, 56]

        enhance3 = self.enhance3(x)
        encoded3 = self.encode3(pool2)
        pool3 = encoded3 + enhance3    # [N, 512, 28, 28]

        enhance4 = self.enhance4(x)
        encoded4 = self.encode4(pool3)
        pool4 = encoded4 + enhance4    # [N, 1024, 14, 14]

        encoded5 = self.encode5(pool4)          # [N, 2048, 14, 14]

        u21 = self.upsample(encoded2)
        u32 = self.upsample(encoded3)
        u43 = self.upsample(enhance4)
        u54 = self.upsample(encoded5)

        d12 = self.downsample(encoded1)
        d23 = self.downsample(encoded2)
        d34 = self.downsample(encoded3)
        d45 = self.downsample(encoded4)

        # print(u21.size(), u32.size(), u43.size(), u54.size())
        # print(d12.size(), d23.size(), d34.size(), d45.size())

        # decode
        fuse5 = torch.cat([encoded5, d45], dim=1)
        # print('fuse5', fuse5.size())
        deconved4 = self.deconv5_4(fuse5)    # [N, 1024, 28, 28]

        fuse4 = torch.cat([encoded4, u54, d34], 1)
        # print('fuse4', fuse4.size(), 'd4', deconved4.size())
        decoded4 = self.decode4(torch.cat([self.se4(fuse4), deconved4], dim=1))      # 512 + 512 -> 512

        deconved3 = self.deconv4_3(decoded4)    # [N, 512, 56, 56]
        fuse3 = torch.cat([encoded3, u43, d23], 1)
        # print('fuse3', fuse3.size(), 'd3', deconved3.size())
        decoded3 = self.decode3(torch.cat([self.se3(fuse3), deconved3], dim=1))

        deconved2 = self.deconv3_2(decoded3)    # [N, 256, 112, 112]
        fuse2 = torch.cat([encoded2, u32, d12], 1)
        # print('fuse2', fuse2.size(), 'd2', deconved2.size())
        decoded2 = self.decode2(torch.cat([self.se2(fuse2), deconved2], dim=1))

        deconved1 = self.deconv2_1(decoded2)    # [N, 64, 224, 224]
        fuse1 = torch.cat([encoded1, u21], 1)
        # print('fuse1', fuse1.size(), 'd1', deconved1.size())
        decoded1 = self.decode1(torch.cat([self.se1(fuse1), deconved1], dim=1))

        output = self.out(self.extra_deconv(decoded1))             # 64 -> 1


        output = F.sigmoid(output)

        return output
