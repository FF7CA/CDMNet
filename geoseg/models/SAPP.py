import torch
import torch.nn as nn
import torch.nn.functional as F

class cSE(nn.Module):
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        z = x.mean(dim=(2, 3), keepdim=True) 
        attention_weights = self.excitation(z)
        return x * attention_weights


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class SAPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, use_depthwise=True):
        super(SAPP, self).__init__()
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        ConvBlock = DepthwiseSeparableConv if use_depthwise else self._standard_conv

        self.conv3x3_1 = ConvBlock(in_channels, out_channels, dilation=6)
        self.conv3x3_2 = ConvBlock(in_channels, out_channels, dilation=12)
        self.conv3x3_3 = ConvBlock(in_channels, out_channels, dilation=18)
        
        self.cse1 = cSE(out_channels, r=16)
        self.cse2 = cSE(out_channels, r=16)
        self.cse3 = cSE(out_channels, r=16)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def _standard_conv(self, in_c, out_c, dilation):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        
        conv1x1 = self.conv1x1(x)
        conv3x3_1 = self.cse1(self.conv3x3_1(x))
        conv3x3_2 = self.cse2(self.conv3x3_2(x))
        conv3x3_3 = self.cse3(self.conv3x3_3(x))

        global_avg = self.global_avg_pool(x)
        global_avg = F.interpolate(global_avg, size=(h, w), mode='bilinear', align_corners=True)
        
        out = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_avg], dim=1)
        out = self.conv_out(out)
        
        return out