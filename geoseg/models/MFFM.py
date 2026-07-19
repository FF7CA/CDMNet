import torch
import torch.nn as nn
import torch.nn.functional as F

class scSE(nn.Module):
    def __init__(self, in_channels, r=16):
        super(scSE, self).__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // r), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // r), in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sse = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return (x * self.cse(x)) + (x * self.sse(x))

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MFFM(nn.Module):
    def __init__(self, in_channels):
        super(MFFM, self).__init__()
        self.scse = scSE(in_channels)
        self.dsconv = DSConv(in_channels, out_channels=1)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1_mid = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_bot = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        spatial_weight = self.sigmoid(self.dsconv(self.scse(x)))
        mid_out = self.conv1x1_mid(x * spatial_weight)
        channel_weight = self.relu(self.conv1x1_bot(self.avg_pool(x)))
        
        return (mid_out * channel_weight) + x

class FeatureFusionNode(nn.Module):
    def __init__(self, in_c_curr, in_c_up=None, in_c_down=None, out_channels=None):
        super(FeatureFusionNode, self).__init__()
        self.out_channels = out_channels or in_c_curr
        
        self.align_curr = nn.Sequential(
            nn.Conv2d(in_c_curr, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        
        if in_c_up is not None:
            self.align_up = nn.Sequential(
                nn.Conv2d(in_c_up, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.align_up = None

        if in_c_down is not None:
            self.align_down = nn.Sequential(
                nn.Conv2d(in_c_down, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.align_down = None

    def forward(self, x_curr, x_up=None, x_down=None):
        target_size = x_curr.shape[2:] 
        
        out = self.align_curr(x_curr)
        
        if self.align_up is not None and x_up is not None:
            feat_up = self.align_up(x_up)
            feat_up = F.interpolate(feat_up, size=target_size, mode='bilinear', align_corners=True)
            out = out + feat_up

        if self.align_down is not None and x_down is not None:
            feat_down = F.adaptive_avg_pool2d(x_down, output_size=target_size)
            feat_down = self.align_down(feat_down)
            out = out + feat_down
            
        return out

class SkipConnectionPyramid(nn.Module):
    def __init__(self, channels):
        super(SkipConnectionPyramid, self).__init__()
        c1, c2, c3, c4 = channels

        self.fusion1 = FeatureFusionNode(in_c_curr=c1, in_c_up=c2, in_c_down=None, out_channels=c1)
        self.mffm1 = MFFM(in_channels=c1)

        self.fusion2 = FeatureFusionNode(in_c_curr=c2, in_c_up=c3, in_c_down=c1, out_channels=c2)
        self.mffm2 = MFFM(in_channels=c2)

        self.fusion3 = FeatureFusionNode(in_c_curr=c3, in_c_up=c4, in_c_down=c2, out_channels=c3)
        self.mffm3 = MFFM(in_channels=c3)

    def forward(self, f1, f2, f3, f4):
        fused_1 = self.fusion1(x_curr=f1, x_up=f2)
        fused_2 = self.fusion2(x_curr=f2, x_up=f3, x_down=f1)
        fused_3 = self.fusion3(x_curr=f3, x_up=f4, x_down=f2)
        
        out_1 = self.mffm1(fused_1)
        out_2 = self.mffm2(fused_2)
        out_3 = self.mffm3(fused_3)
        
        return out_1, out_2, out_3