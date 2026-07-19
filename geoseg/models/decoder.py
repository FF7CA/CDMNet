import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.compress = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.compress(x)

        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, num_classes=2, dim_list=[128, 256, 512, 1024]):
        super(Decoder, self).__init__()
        
        self.decoder4 = DecoderBlock(dim_list[3], dim_list[2])
        self.decoder3 = DecoderBlock(dim_list[2], dim_list[1])
        self.decoder2 = DecoderBlock(dim_list[1], dim_list[0])
        self.decoder1_up = nn.Sequential(
            nn.Conv2d(dim_list[0], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1_conv = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64)
        )

        self.final_up_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_up_spatial = nn.Sequential(
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 32)
        )

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x_c128, x_c256, x_c512, x_c1024):
        x = self.decoder4(x_c1024, x_c512)
        x = self.decoder3(x, x_c256)
        x = self.decoder2(x, x_c128)

        x = self.decoder1_up(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder1_conv(x)

        x = self.final_up_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.final_up_spatial(x)

        x = self.final(x)
        return x