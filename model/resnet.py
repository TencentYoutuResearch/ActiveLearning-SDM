################################################################################################
#                                                                                              #
#             Thanks to the model code from TQS                                                #
#             There is a batch norm layer before last fc layer                                 #
#             github link                                                                      #
#             https://github.com/thuml/Transferable-Query-Selection/blob/main/code/network.py  #
#                                                                                              #
################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import resnet50
import math
import sys
sys.path.append('..')
from torch.autograd import Variable

from util.utils import grad_reverse, GradReverse

# basic block for resnet18 and resnet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# bottleneck for ResNet deeper than resnet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# implement of ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = self.avgpool(out)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out, emb
    
    def get_embedding_dim(self):
        return self.embDim


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def test():
    net = ResNet18()
    y, emb = net(Variable(torch.randn(1,3,32,32)))
    # print(emb.size())


# TQS ResNet
# Attention : a batch norm layer before last fc layer may affect the training process
class ResNet50Fc(nn.Module):

    def __init__(self, bottleneck_dim=256, class_num=100):
        super(ResNet50Fc, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
        self.class_num = class_num
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x, reverse=False, eta=0.1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.bn2(x)
        if reverse:
            x = grad_reverse(x, eta)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.conv1.parameters(), 'lr': lr / 10},
            {'params': self.bn1.parameters(), 'lr': lr / 10},
            {'params': self.maxpool.parameters(), 'lr': lr / 10},
            {'params': self.layer1.parameters(), 'lr': lr / 10},
            {'params': self.layer2.parameters(), 'lr': lr / 10},
            {'params': self.layer3.parameters(), 'lr': lr / 10},
            {'params': self.layer4.parameters(), 'lr': lr / 10},
            {'params': self.avgpool.parameters(), 'lr': lr / 10},
            {'params': self.bottleneck.parameters()},
            # {'params': self.bn2.parameters()},
            {'params': self.fc.parameters()},
        ]

        return parameter_list
