import torch
import torch.nn as nn
import torch.nn.functional as F
from .groupmamba import groupmamba_tiny
import math
from einops import rearrange
from .MFFM import MFFM, SkipConnectionPyramid
from .SAPP import SAPP
from .decoder import Decoder


class CrossDomainBlock(nn.Module):
    def __init__(self, alpha=0.95):
        super(CrossDomainBlock, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None


    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix


    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))


    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))


    def low_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[-alpha_h:, -alpha_w:] = 0
        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_high = self.low_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)

        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh + x

class GroupMambaBackbone(nn.Module):
    def __init__(self):
        super(GroupMambaBackbone, self).__init__()
        self.model = groupmamba_tiny()
        self.cdb1 = CrossDomainBlock()
        self.cdb2 = CrossDomainBlock()
        self.cdb3 = CrossDomainBlock()
        self.cdb4 = CrossDomainBlock()
        
    def forward(self, x):
        def _unpack_to_seq(t):
            if isinstance(t, tuple):
                if len(t) == 3:
                    tensor, H, W = t[0], int(t[1]), int(t[2])
                elif len(t) == 2:
                    a, b = t
                    if isinstance(b, tuple) and len(b) == 2:
                        tensor, H, W = a, int(b[0]), int(b[1])
                    else:
                        raise RuntimeError(f"Unexpected tuple return from module: {t}")
                else:
                    raise RuntimeError(f"Unexpected tuple return from module: len={len(t)}")
            else:
                tensor = t
                if tensor.dim() == 4:
                    H, W = tensor.shape[-2], tensor.shape[-1]
                elif tensor.dim() == 3:
                    B, N, C = tensor.shape
                    s = int(math.isqrt(N))
                    if s * s == N:
                        H, W = s, s
                    else:
                        raise RuntimeError(f"Cannot infer H,W from sequence length N={N}.")
                else:
                    raise RuntimeError(f"Unexpected tensor dim: {tensor.dim()}")

            if tensor.dim() == 4:
                seq = rearrange(tensor, 'b c h w -> b (h w) c')
            else:
                seq = tensor
            return seq, H, W

        def seq_to_img(seq, H, W):
            return rearrange(seq, 'b (h w) c -> b c h w', h=H, w=W)

        # stage1
        out = self.model.patch_embed1(x)
        x_seq, H, W = _unpack_to_seq(out)
        for blk in self.model.block1:
            x_seq = blk(x_seq, H, W)
            x_seq, H, W = _unpack_to_seq(x_seq)
        feat1_seq, H1, W1 = x_seq, H, W

        # stage2
        x_img = seq_to_img(feat1_seq, H1, W1)
        out = self.model.patch_embed2(x_img)
        x_seq, H, W = _unpack_to_seq(out)
        for blk in self.model.block2:
            x_seq = blk(x_seq, H, W)
            x_seq, H, W = _unpack_to_seq(x_seq)
        feat2_seq, H2, W2 = x_seq, H, W

        # stage3
        x_img = seq_to_img(feat2_seq, H2, W2)
        out = self.model.patch_embed3(x_img)
        x_seq, H, W = _unpack_to_seq(out)
        for blk in self.model.block3:
            x_seq = blk(x_seq, H, W)
            x_seq, H, W = _unpack_to_seq(x_seq)
        feat3_seq, H3, W3 = x_seq, H, W

        # stage4
        x_img = seq_to_img(feat3_seq, H3, W3)
        out = self.model.patch_embed4(x_img)
        x_seq, H, W = _unpack_to_seq(out)
        for blk in self.model.block4:
            x_seq = blk(x_seq, H, W)
            x_seq, H, W = _unpack_to_seq(x_seq)
        feat4_seq, H4, W4 = x_seq, H, W


        feat1 = self.cdb1(seq_to_img(feat1_seq, H1, W1))
        feat2 = self.cdb2(seq_to_img(feat2_seq, H2, W2))
        feat3 = self.cdb3(seq_to_img(feat3_seq, H3, W3))
        feat4 = self.cdb4(seq_to_img(feat4_seq, H4, W4))

        return [feat1, feat2, feat3, feat4]
    

class CDMNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CDMNet, self).__init__()
        self.encoder = GroupMambaBackbone()
        self.decoder = Decoder(num_classes, dim_list=[64, 128, 348, 448])
        self.skip_connection = SkipConnectionPyramid(channels=[64, 128, 348, 448])
        self.mffm1 = MFFM(in_channels=64)
        self.mffm2 = MFFM(in_channels=128)
        self.mffm3 = MFFM(in_channels=348)
        self.sapp = SAPP(in_channels=448, out_channels=448, use_depthwise=True)
        
    def forward(self, x):
        features = self.encoder(x)
        out1, out2, out3 = self.skip_connection(features[0], features[1], features[2], features[3])
        out1 = self.mffm1(out1)
        out2 = self.mffm2(out2)
        out3 = self.mffm3(out3)
        features[3] = self.sapp(features[3])
        output = self.decoder(out1, out2, out3, features[3])

        return output
