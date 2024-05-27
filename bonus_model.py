"""Define your architecture here."""
import torch
from models import SimpleNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.stride = stride
        hidden_dim = in_channels * expand_ratio

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.dw_conv(out)))
        out = self.bn3(self.conv2(out))
        if self.use_res_connect:
            return x + out
        else:
            return out

class FixEfficientNetB0(nn.Module):
    # Epochs: 2  Acc: 87.52%
    # Epochs: 10 Acc: 93.28% 
    # num params - 5732546
    def __init__(self, num_classes=2):
        super(FixEfficientNetB0, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(192, 320, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(320, 1280, kernel_size=3, stride=1, expand_ratio=6)
        )
        self.conv2 = nn.Conv2d(1280, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model =  FixEfficientNetB0(num_classes=2) #SimpleNet()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
