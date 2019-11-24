import torch
import torch.nn as nn
from quan_conv import QuanConv as Conv


__all__ = ['ResNet', 'resnet18']



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, midplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet(nn.Module):

    def __init__(self, block, ratio_code, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = int(64*ratio_code[0])

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, int(64*ratio_code[0]), layers[0], ratio_code, 4)
        self.layer2 = self._make_layer(block, 128, int(128*ratio_code[1]), layers[1], ratio_code, 6, stride=2)
        self.layer3 = self._make_layer(block, 256, int(256*ratio_code[2]), layers[2], ratio_code, 8, stride=2)
        self.layer4 = self._make_layer(block, 512, int(512*ratio_code[3]), layers[3], ratio_code, 10, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*ratio_code[3]) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, midplanes, planes, blocks, ratio_code, start, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        j = start
        mid_planes = int(midplanes * ratio_code[j])
        layers = []
        layers.append(block(self.inplanes, mid_planes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            mid_planes = int(midplanes * ratio_code[j+1])
            layers.append(block(self.inplanes, mid_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(ratio_code, **kwargs):
    model = ResNet(BasicBlock, ratio_code, [2, 2, 2, 2])
    return model


