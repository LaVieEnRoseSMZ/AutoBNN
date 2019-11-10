import torch.nn as nn
import torchvision.transforms as transforms
from quan_conv import QuanConv as Conv


class VGG_Cifar10(nn.Module):

    def __init__(self, self, ratio_code, num_classes=10):
        super(VGG_Cifar10, self).__init__()
        in_channels = [3, 128, 128, 256, 256, 512]
        out_channels = [128, 128, 256, 256, 512, 512]
        for i in range(6):
            if i != 5:
                in_channels[i+1] = int(in_channels[i+1]*ratio_code[i])
            out_channels[i] = int(out_channels[i]*ratio_code[i])
        self.in_planes = int(512*4*4*ratio_code[5])
        self.features = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels[0]),

            Conv(in_channels[1], out_channels[1], kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels[1]),

            Conv(in_channels[2], out_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[2]),

            Conv(in_channels[3], out_channels[3], kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels[3]),

            Conv(in_channels[4], out_channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[4]),

            Conv(in_channels[5], out_channels[5], kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels[5]),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.in_planes, 1024, bias=False),
            nn.LogSoftMax()
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.in_planes)
        x = self.classifier(x)
        return x


def vgg_small(ratio_code, num_classes=10, **kwargs):
    return VGG_Cifar10(ratio_code, num_classes)
