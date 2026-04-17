# 视觉RWKV组件定义
from typing import Sequence
import math, os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from einops.layers.torch import Rearrange
from basicsr.module.base.utils.drop import DropPath
import matplotlib.pyplot as plt
# from mmcv.ops.deform_conv import DeformConv2dPack
# from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack, CONV_LAYERS
# from .deconv import DEConv
logger = logging.getLogger(__name__)


T_MAX = 4096 
HEAD_SIZE = 8

from torch.utils.cpp_extension import load

wkv6_cuda = load(name="wkv6",
                 sources=["basicsr/module/base/cuda_v6/wkv6_op.cpp", 
                          "basicsr/module/base/cuda_v6/wkv6_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                 "-O3", "-Xptxas -O3", 
                 "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", 
                 f"-D_T_={T_MAX}"])

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

def q_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE, 
                      patch_resolution=None, with_cls_token=False):
    B, N, C = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.transpose(1, 2).reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1])  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0:int(head_dim):4, :, shift_pixel:W] = \
        input[:, :, 0:int(head_dim):4, :, 0:W-shift_pixel]
    output[:, :, 1:int(head_dim):4, :, 0:W-shift_pixel] = \
        input[:, :, 1:int(head_dim):4, :, shift_pixel:W]
    output[:, :, 2:int(head_dim):4, shift_pixel:H, :] = \
        input[:, :, 2:int(head_dim):4, 0:H-shift_pixel, :]
    output[:, :, 3:int(head_dim):4, 0:H-shift_pixel, :] = \
        input[:, :, 3:int(head_dim):4, shift_pixel:H, :]
    if with_cls_token:
        output = output.reshape(B, C, N-1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N).transpose(1, 2)
    return output

class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)
        
        return out

# class FeatureAdaptiveTheta_v1(nn.Module):
#     """显存优化的特征自适应theta生成器"""
#     def __init__(self, channels):
#         super().__init__()
#         # 全局平均池化
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         # 预测网络
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // 4, 1),
#             nn.Sigmoid()  # 输出[0, 1]
#         )
    
#     def forward(self, x):
#         """
#         Args:
#             x: (B, C, H, W)
#         Returns:
#             theta: (B,) - 返回1D向量而非4D张量
#         """
#         feat = self.gap(x)  # (B, C, 1, 1)
#         feat = feat.view(feat.size(0), -1)  # (B, C)
#         theta = self.fc(feat)  # (B, 1)
#         theta = theta.squeeze(-1)  # (B,) - ✅ 关键：返回1D
#         return theta
# class Conv2d_cd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=0.7):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                               stride, padding, dilation, groups, bias)
        
#         # 轻量级theta生成
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.theta_fc = nn.Linear(in_channels, 1)
#         self.base_theta = nn.Parameter(torch.tensor(theta))
    
#     def forward(self, x):
#         # 生成theta（标量）
#         feat = self.gap(x).flatten(1)
#         theta = torch.sigmoid(self.theta_fc(feat).mean())
#         theta = self.base_theta * theta
        
#         # 标准卷积
#         out = self.conv(x)
        
#         # 融合theta的差分卷积
#         kernel_sum = self.conv.weight.sum(dim=[2, 3], keepdim=True)
#         kernel_diff = kernel_sum * (-theta)
#         out_diff = F.conv2d(x, kernel_diff, bias=None,
#                            stride=self.conv.stride, padding=0,
#                            groups=self.conv.groups)
        
#         # in-place操作
#         out.add_(out_diff)
#         return out
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.theta = theta
        self.theta = nn.Parameter(torch.tensor(theta))
    def forward(self, x):
        out_normal = self.conv(x)

        kernel = self.conv.weight
        kernel_diff = kernel.sum(dim=[2, 3], keepdim=True) * (-self.theta) 
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        out_normal.add_(out_diff)
        return out_normal
        # return out_normal - self.theta * out_diff
    
# class ChannelDiffConv(nn.Module):
#     """
#     通道差分卷积（高效版）
#     计算相邻通道之间的差分，并通过 1x1 卷积映射到输出通道
#     """
#     def __init__(self, in_channels, out_channels, bias=False,theta=0.7):
#         super(ChannelDiffConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # 输入是 C-1 个差分通道
#         self.pwconv = nn.Conv2d(in_channels - 1, out_channels, kernel_size=1, bias=bias)
#         self.theta = theta
#     def forward(self, x):
#         # x: [B, C, H, W]
#         # 相邻通道差分：利用切片而不是循环
#         diff_tensor = x[:, 1:, :, :] - self.theta * x[:, :-1, :, :]  # [B, C-1, H, W]
#         out = self.pwconv(diff_tensor)
#         return out


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def forward(self,x):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        out = nn.functional.conv2d(input=x, weight=conv_weight_ad, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        return out


class Conv2d_hd(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd, self).__init__() 
        self.conv = nn.Conv1d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
            conv_weight_hd[:, :, [0,3,6]] = conv_weight[:, :, :]
            conv_weight_hd[:, :, [2,5,8]] = -conv_weight[:, :, :]
            conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
            # conv_weight_hd = conv_weight_hd.view(conv_shape[0], conv_shape[1], 3, 3)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_hd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff
class Conv2d_vd(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
            conv_weight_vd[:, :, [0,1,2]] = conv_weight[:, :, :]
            conv_weight_vd[:, :, [6,7,8]] = -conv_weight[:, :, :]
            conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
            # conv_weight_hd = conv_weight_hd.view(conv_shape[0], conv_shape[1], 3, 3)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_vd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff
class Conv2d_rd(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__() 
        self.conv = nn.Conv2d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.theta = theta
        self.theta = nn.Parameter(torch.tensor(theta))
    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [1, 2, 3]] = conv_weight[:, :, [0, 1, 2]]
            conv_weight_rd[:, :, [5, 10, 15]] = conv_weight[:, :, [0, 3, 6]]
            conv_weight_rd[:, :, [9, 14, 19]] = conv_weight[:, :, [2, 5, 8]]
            conv_weight_rd[:, :, [21, 22, 23]] = conv_weight[:, :, [6, 7, 8]]
            conv_weight_rd[:, :, [6, 7, 8]] = -conv_weight[:, :, [0, 1, 2]] * self.theta
            conv_weight_rd[:, :, [11, 13]] = -conv_weight[:, :, [3, 5]] * self.theta
            conv_weight_rd[:, :, [16, 17, 18]] = -conv_weight[:, :, [6, 7, 8]] * self.theta
            # conv_weight_rd[:, :, 12] = conv_weight[:, :, 4] * (1 - self.theta)
            



            # conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            # conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            # conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
            return out_diff


class Conv2d_lp(nn.Module):
    def __init__(self,  out_channels, in_channels,
                 stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d( out_channels, in_channels, kernel_size=1,
                             stride=1, padding=0, dilation=1, groups=groups, bias=False)

    def forward(self, x):

        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_lp = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)

        conv_weight_lp[:, :, [1,3,5,7]] = conv_weight.squeeze(-1)
        conv_weight_lp[:, :, 4] = -4 * conv_weight.squeeze(-1).squeeze(-1)
        conv_weight_lp = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=3, k2=3)(conv_weight_lp)
        out_diff = nn.functional.conv2d(input=x, weight=conv_weight_lp, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=self.conv.groups)

        return out_diff

# class MultiConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, theta=1.0):
#         super(MultiConvBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.theta = theta

#         # ===== 分支定义 =====
#         self.lpconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False)
#         self.rdconv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=1, bias=False)
#         self.hdconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False)
#         self.vdconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False)

#         # 权重融合参数
#         self.alpha = nn.Parameter(torch.ones(5), requires_grad=True)  # [原输入, lp, rd, hd, vd]

#         # ===== 用于推理的单一卷积 =====
#         self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=1, bias=False)
#         self.reparam_done = False

#     def forward_train(self, x):
#         lp = self.lpconv(x)
#         rd = self.rdconv(x)
#         hd = self.hdconv(x)
#         vd = self.vdconv(x)

#         out = (
#             self.alpha[0] * x +
#             self.alpha[1] * lp +
#             self.alpha[2] * rd +
#             self.alpha[3] * hd +
#             self.alpha[4] * vd
#         )
#         return out, self.alpha[0].detach()

#     # ----------------------------
#     # 权重融合（自动在第一次eval时调用）
#     # ----------------------------
#     def reparam(self):
#         with torch.no_grad():
#             # 各分支权重取出并补齐到5x5大小
#             id_weight = F.pad(torch.ones_like(self.lpconv.weight), (1, 1, 1, 1))
#             lp_weight = F.pad(self.lpconv.weight, (1, 1, 1, 1))
#             rd_weight = self.rdconv.weight
#             hd_weight = F.pad(self.hdconv.weight, (1, 1, 1, 1))
#             vd_weight = F.pad(self.vdconv.weight, (1, 1, 1, 1))

#             W_eff = (
#                 self.alpha[0] * id_weight +
#                 self.alpha[1] * lp_weight +
#                 self.alpha[2] * rd_weight +
#                 self.alpha[3] * hd_weight +
#                 self.alpha[4] * vd_weight
#             )

#             self.reparam_conv.weight.copy_(W_eff)
#             self.reparam_done = True

#             del self.lpconv, self.rdconv, self.hdconv, self.vdconv, self.alpha
#             torch.cuda.empty_cache()


#     def forward_reparam(self, x):
#         out = self.reparam_conv(x)
#         return out, getattr(self, 'alpha', torch.tensor([1.0], device=x.device))[0].detach()
#     def forward(self, x):
#         if self.training:
#             return self.forward_train(x)

#         if not self.reparam_done:
#             self.reparam()
#         return self.forward_reparam(x)
    

# class Conv2d_rd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=2, dilation=1, bias=False, theta=1.0):
#         super(Conv2d_rd, self).__init__()
#         self.theta = theta
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Step 1: Depthwise 5x5 conv (groups=in_channels)
#         self.dwconv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=5, stride=stride,
#             padding=padding, dilation=dilation, groups=in_channels, bias=False
#         )

#         # Step 2: Pointwise 1x1 conv
#         self.pwconv = nn.Conv2d(
#             in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
#         )

#         # 初始化 depthwise 卷积核为差分核
#         self.reset_parameters()

#     def reset_parameters(self):
#         """把DWConv初始化为差分卷积核"""
#         with torch.no_grad():
#             for c in range(self.in_channels):
#                 # 初始化为 5x5 零
#                 kernel = torch.zeros((5, 5))
#                 # 中心点
#                 kernel[2, 2] = (1 - self.theta)
#                 # 邻域正权重
#                 kernel[0, 0] = kernel[0, 2] = kernel[0, 4] = 1
#                 kernel[2, 0] = kernel[2, 4] = 1
#                 kernel[4, 0] = kernel[4, 2] = kernel[4, 4] = 1
#                 # 邻域负权重
#                 kernel[1, 2] = kernel[2, 1] = kernel[2, 3] = kernel[3, 2] = -self.theta

#                 self.dwconv.weight.data[c, 0] = kernel

#     def forward(self, x):
#         out = self.dwconv(x)   # depthwise (差分卷积)
#         out = self.pwconv(out) # pointwise (跨通道融合)
#         return out
# class DPNet(nn.Module):
#     def __init__(self, out_channels, in_channels, kernel_size=3, stride=1,
#                 dilation=1, groups=1, bias=False, theta=1.0):

#         super(DPNet, self).__init__() 
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=in_channels, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=1, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=2, groups=1, bias=False)
#         self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

#         self.theta = theta
#         self.conv_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, groups=1, bias = False) 
#         self.repram_flag = True

#         # self.out1_weight = nn.Parameter(torch.zeros_like(self.conv_reparam.weight), requires_grad=False)
#         # self.out2_weight = nn.Parameter(torch.zeros_like(self.conv2.weight), requires_grad=False)  # 用于存储deconv的权重
#         # self.out3_weight = nn.Parameter(torch.zeros_like(self.conv_reparam.weight), requires_grad=False)
#         self.register_buffer('out1_weight', torch.zeros_like(self.conv_reparam.weight), persistent=True)
#         self.register_buffer('out2_weight', torch.zeros_like(self.conv2.weight), persistent=True)
#         self.register_buffer('out3_weight', torch.zeros_like(self.conv_reparam.weight), persistent=True)

#     def deconv(self, x):
#         out2 = self.conv2(x)
#         kernel = self.conv2.weight
#         kernel_diff = kernel.sum(dim=[2, 3], keepdim=True)  # 形状：(C_out, C_in, 1, 1)
#         kernel_c = kernel.clone()
#         kernel_c[ :, :, 1,1] = kernel_c[:, :, 1,1] - 0.7*kernel_diff[:, :, 0, 0] 
#         # out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv2.bias, stride=self.conv2.stride, padding=0, groups=self.conv2.groups)
#         # out2 = out2 - self.theta * out_diff
#         # self.out2_weight  = nn.Parameter(out2)
#         with torch.no_grad():
#             self.out2_weight.copy_(kernel_c)
#         # self.out2_weight = kernel_c
#         out2 = nn.functional.conv2d(input=x, weight=self.out2_weight, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)
#         return out2

#     def rdconv(self, x):
#         conv_weight = self.conv3.weight
#         conv_shape = conv_weight.shape

#         # conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
#         if conv_weight.is_cuda:
#             conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
#         else:
#             conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
#         conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
#         conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
#         conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
#         conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
#         conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
#         with torch.no_grad():
#             self.out3_weight.copy_(conv_weight_rd)
#         out3 = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding, groups=self.conv3.groups)
#         return out3
#     def forward_train(self, x):
#         out1 = self.conv1(x)
#         out2 = self.deconv(x)
#         out3 = self.rdconv(x)
#         out = self.alpha[0]*x + self.alpha[1]*out1 + self.alpha[2]*out2 + self.alpha[3]*out3
#         return out

#     def reparam(self):
        
#         padded_weight_1 = F.pad(self.conv1.weight, (2, 2, 2, 2)) 
#         for c in range(self.out1_weight.shape[0]):
#             self.out1_weight[c, c] = padded_weight_1[c,0]

#         padded_weight_2 = F.pad(self.out2_weight, (1, 1, 1, 1))
#         identity_weight = F.pad(torch.ones_like(self.conv1.weight), (2, 2, 2, 2)) 
#         combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*self.out1_weight + self.alpha[2]*self.out2_weight + self.alpha[3]*self.out3_weight
#         device = self.conv_reparam.device 
#         combined_weight = combined_weight.to(device)
#         self.conv_reparam.weight = nn.Parameter(combined_weight)
#     def forward(self, x): 
        
#         if self.training: 
#             self.repram_flag = True
#             out = self.forward_train(x) 
#         elif self.training == False and self.repram_flag == True:
#             self.reparam() 
#             self.repram_flag = False 
#             out = self.conv_reparam(x)
#         elif self.training == False and self.repram_flag == False:
#             out = self.conv_reparam(x)
        
#         return out
        
# class Conv2d_hd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=1.0):

#         super(Conv2d_hd, self).__init__() 
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

#     def get_weight(self):
#         conv_weight = self.conv.weight
#         conv_shape = conv_weight.shape
#         conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
#         conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
#         conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
#         conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
#         return conv_weight_hd, self.conv.bias


# class Conv2d_vd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False):

#         super(Conv2d_vd, self).__init__() 
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
#     def get_weight(self):
#         conv_weight = self.conv.weight
#         conv_shape = conv_weight.shape
#         conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
#         conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
#         conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
#         conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
#         return conv_weight_vd, self.conv.bias


# class DEConv(nn.Module):
#     def __init__(self, dim):
#         super(DEConv, self).__init__() 
#         self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
#         # self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
#         # self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
#         # self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
#         # self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

#     def forward(self, x):
#         w1, b1 = self.conv1_1.get_weight()
#         # w2, b2 = self.conv1_2.get_weight()
#         # w3, b3 = self.conv1_3.get_weight()
#         # w4, b4 = self.conv1_4.get_weight()
#         # w5, b5 = self.conv1_5.weight, self.conv1_5.bias

#         w = w1
#         b = b1
#         res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

#         return res

# class KernelSelectiveFusionAttention(nn.Module):
#     def __init__(self, dim, r=16, L=32):
#         super().__init__()
#         d = max(dim // r, L)
#         self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
#         self.conv1 = nn.Conv2d(dim, dim // 2, 1)
#         self.conv2 = nn.Conv2d(dim, dim // 2, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv = nn.Conv2d(dim // 2, dim, 1)
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.global_maxpool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(dim, d, 1, bias=False),
#             nn.BatchNorm2d(d),
#             nn.ReLU(inplace=True)
#         )
#         self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#     def forward(self, x):
#         batch_size = x.size(0)
#         dim = x.size(1)
#         attn1 = self.conv0(x)  # conv_3*3
#         attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5
#         attn1 = self.conv1(attn1) # b, dim/2, h, w
#         attn2 = self.conv2(attn2) # b, dim/2, h, w
#         attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w
#         avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w
#         max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w
#         agg = torch.cat([avg_attn, max_attn], dim=1) # spa b,2,h,w
#         ch_attn1 = self.global_pool(attn) # b,dim,1, 1
#         z = self.fc1(ch_attn1)
#         a_b = self.fc2(z)
#         a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
#         a_b = self.softmax(a_b)
#         a1,a2 =  a_b.chunk(2, dim=1)
#         a1 = a1.reshape(batch_size,dim // 2,1,1)
#         a2 = a2.reshape(batch_size, dim // 2, 1, 1)
#         w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
#         w2 = a2 * agg[:, 0, :, :].unsqueeze(1)
#         attn = attn1 * w1 + attn2 * w2
#         attn = self.conv(attn).sigmoid()
#         return x * attn
# class LaplaceConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
#         super(LaplaceConv2d, self).__init__()
        
#         # 定义一个固定值的 3x3 Laplace 卷积核
#         self.laplace_kernel = torch.tensor([[0.0, -1.0, 0.0],
#                                             [-1.0, 4.0, -1.0],
#                                             [0.0, -1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#         # 扩展卷积核的形状以匹配输入和输出通道
#         self.laplace_kernel = self.laplace_kernel.expand(out_channels, in_channels, kernel_size, kernel_size)

#         # 定义卷积层
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
#         # 手动设置卷积层的权重为 laplace_kernel
#         self.conv.weight = nn.Parameter(self.laplace_kernel)

#     def forward(self, x):
#         # 使用 nn.Conv2d 进行卷积操作
#         return self.conv(x)


# def save_single_heatmap(B, C, layer_id, data, name):
#     save_path = '/data/model/tql/visualizations'
#     os.makedirs(save_path, exist_ok=True)  # 确保目录存在
#     for i in range(B):
#         for c in range(C):
#             fig, ax = plt.subplots(figsize=(3, 3))
#             ax.imshow(data[i, c, :, :],  cmap='viridis', interpolation='nearest')
#             ax.axis('off')  # 不显示坐标轴
#             plt.tight_layout()
            
#             # 使用 layer_id 和 c 来区分每个图像
#             file_name = f'{save_path}/{layer_id}_{name}_heatmap_{i}_channel_{c}.png'
#             plt.savefig(file_name)
#             print(f'Saved heatmap: {file_name}')
#             plt.close()
class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(self, n_embd,embed, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, init_mode='fancy', key_norm=False, with_cls_token=False, 
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        # self.shift_pixel = shift_pixel
        # self.shift_mode = shift_mode
        # self.shift_func = eval(shift_mode)

        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # if key_norm:
        #     self.key_norm = nn.LayerNorm(n_embd)
        # else:
        #     self.key_norm = None
        self.output = nn.Linear(self.attn_sz, self.n_embd, bias=False)
        self.output.init_scale = 0

        self.ln_x = nn.LayerNorm(self.attn_sz)
        self.with_cp = with_cp
        # self.omni_shift = OmniShift(dim=n_embd)
        # self.dpnet = DPNet(out_channels=n_embd, in_channels=n_embd, kernel_size=3, stride=1,dilation=1, groups=1, bias=False, theta=0.7)
        # self.conv1x1 = nn.Conv2d(in_channels=self.n_embd, out_channels=self.n_embd, kernel_size=1, groups=self.n_embd, bias=False)
        # self.conv3x3 = nn.Conv2d(in_channels=2*self.n_embd, out_channels=self.n_embd, kernel_size=3, padding=1, groups=self.n_embd, bias=False)
        self.lpconv = Conv2d_lp(in_channels=self.n_embd, out_channels=self.n_embd, stride=1, padding=1, groups=1, bias=False)
        self.hdconv = Conv2d_hd(self.n_embd, self.n_embd, 3, bias=False,groups=1, theta=0.7)
        self.vdconv = Conv2d_vd(self.n_embd, self.n_embd, 3, bias=False,groups=1, theta=0.7)
        # self.inhdconv = Conv2d_hd(self.n_embd, self.n_embd, 3, bias=False,groups=1, theta=0.7)
        # self.invdconv = Conv2d_vd(self.n_embd, self.n_embd, 3, bias=False,groups=1, theta=0.7)
        self.deconv = Conv2d_cd(self.n_embd, self.n_embd, 3, bias=False,groups=1, theta=0.7)
        self.rdconv = Conv2d_rd(in_channels=self.n_embd, out_channels=self.n_embd, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=False, theta=0.7)
        # self.adconv = Conv2d_ad(in_channels=self.n_embd, out_channels=self.n_embd, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7)
        self.alpha = nn.Parameter(torch.randn(5), requires_grad=True) 
        # self.laplance = LaplaceConv2d(in_channels=self.n_embd, out_channels=self.n_embd, kernel_size=3, stride=1, padding=1, bias=False)
        # self.num_clusters = 5
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.score_net = ViTScoreMap(
        #     in_channels=self.n_embd,
        #     patch_size=16,
        #     embed_dim=self.n_embd,
        #     num_layers=1,
        #     num_heads=4
        # ).to(device)
        # self.score_map_patch_size = 16
        # self.gamma = nn.Parameter(torch.rand(1))
        # self.multiconv = MultiConvBlock(in_channels=self.n_embd, out_channels=self.n_embd, theta=0.7)
        # self.ln = nn.LayerNorm(n_embd)
        # self.proj = nn.Linear(n_embd, n_embd)
        # self.deconv =  DEConv(self.n_embd)
        # self.conv3 = nn.Conv2d(in_channels=self.n_embd, out_channels=self.n_embd, kernel_size=3, padding=1, groups=self.n_embd, bias=False) 
        # self.conv5 = nn.Conv2d(in_channels=self.n_embd, out_channels=self.n_embd, kernel_size=5, padding=2, groups=self.n_embd, bias=False)
        # self.act1 = nn.ReLU(inplace=True)
        # self.ksnet = KernelSelectiveFusionAttention(dim=n_embd, r=16, L=32)
        # self.cd_conv = Conv2d_cd(n_embd,n_embd,kernel_size=3, stride=1, padding=1, bias=False, theta= 0.7)
        # self.fgem = FGEM(embed_dim = n_embd,in_chans = n_embd, n_groups=embed)
        # self.conv_scale = nn.Parameter(torch.ones(n_embd))
        # self.conv1b3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv1a3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv33 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1, bias=False,
		# 			  groups=n_embd),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.SiLU(),
		# )
        

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    ddd[0, 0, i] = i / self.n_embd

                # fancy time_mix
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                # self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                # self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                # self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                # self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                # self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                # self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                # TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
                # self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
                # self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-1e-4, 1e-4))

                # fancy time_decay
                # decay_speed = torch.ones(self.attn_sz)
                # for n in range(self.attn_sz):
                #     decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                # self.time_decay1 = nn.Parameter(decay_speed.reshape(1,1,self.attn_sz))
                # self.time_decay2 = nn.Parameter(decay_speed.reshape(1,1,self.attn_sz))
                self.time_decay1 = nn.Parameter(torch.randn(1, 1, self.attn_sz)) 
                # self.time_decay2 = nn.Parameter(torch.randn(1, 1, self.attn_sz))

                TIME_DECAY_EXTRA_DIM = 64
                # self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM))
                # self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-2, 1e-2))
                self.time_decay_w1_1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                self.time_decay_w1_2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-4, 1e-4))

                # self.time_decay_w2_1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                # self.time_decay_w2_2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-4, 1e-4))

                # tmp = torch.zeros(self.attn_sz)
                # for n in range(self.attn_sz):
                #     zigzag = ((n + 1) % 3 - 1) * 0.1
                #     tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa_1 = nn.Parameter(torch.randn(self.n_head, self.head_size)) 
                self.time_faaaa_2 = nn.Parameter(torch.randn(self.n_head, self.head_size))
                # self.time_faaaa_3 = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
                # self.time_faaaa_4 = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

                # self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError
        
    def jit_func(self, x):
        # if x.dim() == 4:
        #     h, w = patch_resolution
        #     x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w).contiguous()


        # Mix x with the previous timestep to produce xk, xv, xr
        # B, T, C = x.size()
        # h, w = patch_resolution
        # xx = rearrange(x, 'b (h w) c -> b c h w', h=h,w=w)
        
        # xx = self.conv1b3(self.conv33(self.conv1a3(xx)))
        
        # xx = self.fgem(xx)  # [B, C, H, W]
        # xx = self.omni_shift(xx)
        # xx = self.ksnet(xx)  # [B, C, H, W]
        # xx = rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w) *  self.conv_scale + x
        # xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution, 
        #                      with_cls_token=self.with_cls_token) - x

        xw = x + x * (self.time_maa_w)
        xk = x + x * (self.time_maa_k)
        xv = x + x * (self.time_maa_v)
        xr = x + x * (self.time_maa_r)
        xg = x + x * (self.time_maa_g)

        # xw = x
        # xk = x
        # xv = x
        # xr = x
        # xg = x


        # xxx = x + xx * self.time_maa_x  # [B, T, C]
        # xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        # # [5, B*T, TIME_MIX_EXTRA_DIM]
        # xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        # # [5, B, T, C]
        # mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # xw = x + xx * (self.time_maa_w + mw)
        # xk = x + xx * (self.time_maa_k + mk)
        # xv = x + xx * (self.time_maa_v + mv)
        # xr = x + xx * (self.time_maa_r + mr)
        # xg = x + xx * (self.time_maa_g + mg)

        # xr, xk, xv, xw = x, x, x, x

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))
        
        # ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # # [B, T, C]
        # w = self.time_decay + ww
        ww1 = torch.tanh(xw @ self.time_decay_w1_1) @ self.time_decay_w1_2 # [B, T, C]
        w1 = self.time_decay1 + ww1

        # ww2 = torch.tanh(xw @ self.time_decay_w2_1) @ self.time_decay_w2_2 # [B, T, C]
        # w2 = self.time_decay2 + ww2

        # return r, k, v, g, w1,w2
        return r, k, v, g, w1

    def jit_func_2(self, x, g):
        x = self.ln_x(x)
        x = self.output(x*g)
        return x
    # def jit_func_2(self, x, g):
    #     B, T, C = x.size()
    #     x = x.view(B * T, C)
        
    #     x = self.ln_x(x).view(B, T, C)
    #     x = self.output(x * g)
    #     return x
    # def rwkv(self,x,patch_resolution,B, T, C):
    #     h, w = patch_resolution
    #     x = rearrange(x, 'b c h w -> b (h w) c', h=h,w=w) 
    #     r, k, v, g, w1, w2 = self.jit_func(x, patch_resolution)
    #     if self.layer_id%2==0:
    #         x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_1)
    #     else:
    #         h, w = patch_resolution
    #         r = rearrange(r, 'b (h w) c -> b (w h) c', h=h, w=w) 
    #         k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
    #         v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
    #         v = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w2, u=self.time_faaaa_2)
    #         x = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
    #     return self.jit_func_2(x,g)

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device
            h, w = patch_resolution
            # H_patch, W_patch = h // self.score_map_patch_size, w // self.score_map_patch_size
            # N_patches = H_patch * W_patch
            xx = rearrange(x, 'b (h w) c -> b c h w', h=h,w=w)
            
            # xxx = self.deconv(xx)
            # xx = self.act1(xxx)+xx
            # xx = self.conv1x1(xx)

            # xx = self.rwkv(xx,patch_resolution,B, T, C)

            # xx1,xx2,xx3,xx4 = xx.chunk(4, dim=1)  # [B, C/4, H, W] * 4

            # # x1 = self.rwkv(xx1,patch_resolution,B, T, C//4)
            # xx1 = self.alpha[0]*self.conv1x1(xx1)
            # # x2 = self.rwkv(xx2,patch_resolution,B, T, C//4)
            # xx2 = self.alpha[1]*self.deconv(xx2)
            # # xx3 = self.hdconv(xx3)+xx3
            # xx3 = self.alpha[2]*self.conv3(xx3)

            # xx4 = self.alpha[3]*self.conv5(xx4)
            # # # x3 = self.rwkv(xx3,patch_resolution,B, T, C//4)
            # # xx4 = self.vdconv(xx4)+xx4
            # # # x4 = self.rwkv(xx4,patch_resolution,B, T, C//4)
            # xx = torch.cat([xx1, xx2, xx3, xx4], dim=0) 
            # self.deconv(xx2)
            # xx = self.rwkv(xx,patch_resolution,B*4, T, C//4)
            # xx = rearrange(xx, ' (b4 b) (h w) c -> b (h w) (b4 c)', b4=4, h=h,w=w, b=B)
            # return xx
            
            de = self.deconv(xx)  # [B, C, H, W]
            # ad = self.adconv(xx)  # [B, C, H, W]
            rd = self.rdconv(xx)  # [B, C, H, W]
            lp = self.lpconv(xx)  # [B, C, H, W]
            # lp_origin = self.laplance(xx)  # [B, C, H, W]
            # print('Visualizing feature maps...')
            # lp_origin_cpu = lp_origin.detach().cpu().numpy()
            
            # save_single_heatmap(B,C,self.layer_id,lp_origin_cpu, 'lp_origin')


            # de_cpu = de.detach().cpu().numpy()
            # rd_cpu = rd.detach().cpu().numpy()
            # lp_cpu = lp.detach().cpu().numpy()
            
            # save_single_heatmap(B,C,self.layer_id,de_cpu, 'de')
            # save_single_heatmap(B,C,self.layer_id,rd_cpu, 'rd')
            # save_single_heatmap(B,C,self.layer_id,lp_cpu, 'lp')

            # out1x1= self.conv1x1(xx)  # [B, C, H, W]

            # hd = self.hdconv(xx)  # [B, C, H, W]
            # vd = self.vdconv(xx)  # [B, C, H, W]
            # inhd = self.inhdconv(xx)  # [B, C, H, W]
            # invd = self.invdconv(xx)  # [B, C, H, W]
        
            # conv5 = self.conv5(xx)  # [B, C, H, W]
            # xx = self.alpha[0]*hd + self.alpha[1]*out1x1 + self.alpha[2]*de + self.alpha[3]*rd+ self.alpha[4]*vd
            # xx = rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w) 
                
            # r, k, v, g, w1, w2 = self.jit_func(xx, patch_resolution)
            # # xx = torch.cat([xx1, out1x1, de, rd], dim=1)  # [B, C, H, W]
            # xx = self.alpha[0]*hd + self.alpha[1]*out1x1 + self.alpha[2]*de + self.alpha[3]*rd
            # # # xx = self.omni_shift(xx)  # [B, C, H, W]

            # # # xx = self.conv1b3(self.conv33(self.conv1a3(xx)))
            # # # xx = self.dpnet(xx)  # [B, C, H, W]

            # xx = rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w) 
            # # xx1,xx2,xx3,xx4 = xx.chunk(4, dim=2)
            # # *  self.conv_scale + x
            # r, k, v, g, w1, w2 = self.jit_func(xx, patch_resolution)
            # # r, k, v, g, w = self.jit_func(x, patch_resolution)
            xx = self.alpha[0]*xx + self.alpha[1]*lp +self.alpha[2]*de + self.alpha[3]*rd

            # xx,alpha = self.multiconv(xx)

            # xx1 = rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w) 
                
            # r, k, v, g, w1,w2 = self.jit_func(xx1)

            # x_out = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_1)


            # score_maps,H_patch,W_patch = self.score_net(xx)
            # scan_orders = [adaptive_patch_traversal(s_map) for s_map in score_maps]
            # xs = self.forward_core_adaptive(xx, scan_orders,H_patch,W_patch)
            # N_patches = H_patch * W_patch
            # # patch_resolution_resized = (xs.shape[3], xs.shape[3])
            # # --- 修复维度问题 ---
            # if xs.dim() == 4:
            #     # [B, 1, L, C] -> [B, L, C]
            #     xs = xs.squeeze(1)
            # elif xs.dim() == 2:
            #     # [L, C] -> [1, L, C]（极端情况）
            #     xs = xs.unsqueeze(0)
            # x_seq = xs.permute(0, 2, 1).contiguous()
            # r, k, v, g, w1,w2 = self.jit_func(x_seq)
            # # xx = rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w) 
            # y = RUN_CUDA_RWKV6(B, N_patches, C, self.n_head, r, k, v, w2, u=self.time_faaaa_2)
            # out_y = self.jit_func_2(y,g)
            # unshuffled_y = torch.zeros_like(out_y)
            # for b in range(B):
            #     order_indices = torch.tensor([i * W_patch + j for i, j in scan_orders[b]], device=x.device, dtype=torch.long)
            #     inverse_order = torch.empty_like(order_indices)
            #     inverse_order[order_indices] = torch.arange(N_patches, device=x.device)
            #     unshuffled_y[b] = out_y[b, inverse_order, :]
            # # y_fused = unshuffled_y[:, 0]
            # y_map = unshuffled_y.permute(0, 2, 1).contiguous().view(B, C, H_patch, W_patch)
            # y_reconstructed = F.interpolate(y_map, size=(h, w), mode='nearest')
            # y = rearrange(y_reconstructed, 'b c h w -> b (h w) c', h=h,w=w) 

            # return (1 - self.gamma) * y + self.gamma * x_out
            # return y

        
            #### 计算融合后的相似度分数 ##########
            # xc = xx.mean(dim=1, keepdim=True)
            # sim_global = torch.cosine_similarity(xx, xc, dim=-1)

            # idx = torch.argsort(sim, dim=-1, descending=True)
            # k_reordered = torch.gather(k, 1, idx.unsqueeze(-1).expand_as(k))

    

            # r, k, v, g, w1, w2 = self.jit_func(xx, patch_resolution)
            # kc = k.mean(dim=1, keepdim=True)
            # sim_k = torch.cosine_similarity(k, kc, dim=-1)
            
            # H, W = patch_resolution
            # eps = 1e-6
            # alpha = 0.6
            # sigma = 0.25     # 控制高斯半径
            # beta = 0.7       # 控制语义 vs 位置权重
            # x_encoded = k
            # # ---------- (3.2) 局部语义一致性（3x3平均相似度） ----------
            # x_map = rearrange(x_encoded, 'b (h w) c -> b c h w', h=H, w=W)
            # k_local = F.avg_pool2d(x_map, kernel_size=3, stride=1, padding=1)
            # num = (x_map * k_local).sum(dim=1)
            # den = (x_map.norm(dim=1) * k_local.norm(dim=1)).clamp_min(eps)
            # s_local = num / den
            # s_local = (s_local - s_local.amin(dim=(1,2), keepdim=True)) / (
            #     s_local.amax(dim=(1,2), keepdim=True) - s_local.amin(dim=(1,2), keepdim=True) + eps
            # )
            # s_local_flat = rearrange(s_local, 'b h w -> b (h w)')

            # # ---------- (3.3) PCA 主方向得分 ----------
            # x_centered = x_encoded - x_encoded.mean(dim=1, keepdim=True)
            # cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (x_centered.size(1) - 1)
            # eigvals, eigvecs = torch.linalg.eigh(cov)
            # v1 = eigvecs[:, :, -1:].contiguous()
            # proj = torch.bmm(x_centered, v1).squeeze(-1)
            # s_pca = (proj - proj.amin(dim=1, keepdim=True)) / (
            #     proj.amax(dim=1, keepdim=True) - proj.amin(dim=1, keepdim=True) + eps
            # )

            # # ---------- (3.4) 融合语义分数 (local + pca) ----------
            # sim_sem = alpha * s_local_flat + (1 - alpha) * s_pca
            # sim_sem = (sim_sem - sim_sem.amin(dim=1, keepdim=True)) / (
            #     sim_sem.amax(dim=1, keepdim=True) - sim_sem.amin(dim=1, keepdim=True) + eps
            # )

            # # 构造归一化坐标 [0,1]
            # yy, xx = torch.meshgrid(
            #     torch.linspace(0, 1, H, device=k.device),
            #     torch.linspace(0, 1, W, device=k.device),
            #     indexing='ij'
            # )
            # coords = torch.stack([xx, yy], dim=-1).view(1, H*W, 2).expand(B, -1, -1)  # [B,T,2]

            # # 用语义分数计算“语义中心”
            # w = sim_sem + 1e-6
            # center = (coords * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)  # [B,2]

            # # 距离 & 高斯加权
            # dist2 = ((coords - center.unsqueeze(1)) ** 2).sum(dim=-1)  # [B,T]
            # sim_pos = torch.exp(-dist2 / (2 * sigma * sigma))

            # # 归一化
            # sim_pos = (sim_pos - sim_pos.amin(dim=1, keepdim=True)) / (
            #     sim_pos.amax(dim=1, keepdim=True) - sim_pos.amin(dim=1, keepdim=True) + eps
            # )

            # # ---------- (融合语义 + 位置) ----------
            # sim = beta * sim_sem + (1 - beta) * sim_pos
            # sim = (sim - sim.amin(dim=1, keepdim=True)) / (
            #     sim.amax(dim=1, keepdim=True) - sim.amin(dim=1, keepdim=True) + eps
            # )

            # # # idx = torch.argsort(sim, dim=-1, descending=True)
            # # # inv_idx = torch.argsort(idx, dim=-1)
            # # sorted_sim, sorted_idx = torch.sort(sim, dim=-1, descending=True)
            # # tokens_per_cluster = T // self.num_clusters
            # # group_ids = torch.arange(num_clusters, device=x.device).repeat_interleave(tokens_per_cluster)
            # # if group_ids.numel() < T:
            # #     group_ids = torch.cat([group_ids, group_ids.new_full((T - group_ids.numel(),), num_clusters - 1)])
            # # group_ids = group_ids.unsqueeze(0).expand(B, -1)  # [B, T]
            # # H, W = patch_resolution
            # num_groups = 4  # 分几组，可调

            # # ---------- (3) 等量分组 ----------
            # sorted_sim, sorted_idx = torch.sort(sim, dim=-1, descending=True)
            # tokens_per_group = T // num_groups

            # # 为每个 token 构造 group_id
            # group_ids = torch.arange(num_groups, device=k.device).repeat_interleave(tokens_per_group)
            # if group_ids.numel() < T:
            #     group_ids = torch.cat([group_ids, group_ids.new_full((T - group_ids.numel(),), num_groups - 1)])
            # group_ids = group_ids.unsqueeze(0).expand(B, -1)  # [B, T]

            # # ---------- (4) 构造位置索引 ----------
            # pos_y = torch.arange(H, device=k.device).repeat_interleave(W)
            # pos_x = torch.arange(W, device=k.device).repeat(H)
            # pos_id = pos_y * W + pos_x  # [T], 行优先扫描
            # pos_idx = pos_id.unsqueeze(0).expand(B, -1)  # [B, T]

            # # ---------- (5) 向量化排序：先 group 再位置 ----------
            # sorted_pos = torch.gather(pos_idx, 1, sorted_idx)  # [B, T]
            # # 组合排序键（先 group，再位置）
            # key = group_ids * (H * W + 1) + sorted_pos
            # _, final_order = torch.sort(key, dim=-1)

            # # 最终索引顺序
            # final_idx = torch.gather(sorted_idx, 1, final_order)

            # # ---------- (6) 根据最终顺序重排 r, k, v ----------
            # expand_idx = final_idx.unsqueeze(-1).expand(-1, -1, C)
            # r_reordered = torch.gather(r, 1, expand_idx)
            # k_reordered = torch.gather(k, 1, expand_idx)
            # v_reordered = torch.gather(v, 1, expand_idx)

            # B, T = final_idx.shape
            # inverse_idx = torch.empty_like(final_idx)
            # for b in range(B):
            #     inverse_idx[b].scatter_(0, final_idx[b], torch.arange(T, device=final_idx.device))

            # # flip_idx = torch.flip(idx, dims=[-1])
            # # inv_flip_idx = torch.argsort(flip_idx, dim=-1)

            # # k_reordered = torch.gather(k, 1, idx.unsqueeze(-1).expand_as(k))
            # # v_reordered = torch.gather(v, 1, idx.unsqueeze(-1).expand_as(v))
            # # r_reordered = torch.gather(r, 1, idx.unsqueeze(-1).expand_as(r))

            # if self.layer_id%2==0:
            # x_sorted = RUN_CUDA_RWKV6(B, T, C, self.n_head, r_reordered, k_reordered, v_reordered, w1, u=self.time_faaaa_1)
            # expand_inv = inverse_idx.unsqueeze(-1).expand(-1, -1, C)
            # x = torch.gather(x_sorted, 1, expand_inv)

            # x = torch.gather(x_sorted, 1, inv_idx.unsqueeze(-1).expand_as(x_sorted))
            # else:
                

            if self.layer_id%4==0:
                # xx = torch.cat([xx,self.hdconv(xx)], dim=1)
                # xx = self.conv1x1(self.conv3x3(xx))  
                xx = xx + self.alpha[4]*self.hdconv(xx)

                r, k, v, g, w1 = self.jit_func(rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w))

                x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_1)
            

            elif self.layer_id%4==1:
                h, w = patch_resolution
                # xx = torch.cat([xx,self.vdconv(xx)], dim=1)
                # xx = self.conv1x1(self.conv3x3(xx))  
                xx = xx + self.alpha[4]*self.vdconv(xx)
                xx3 = rearrange(xx, 'b c h w -> b (h w) c', h=h,w=w) 
                
                r, k, v, g, w1 = self.jit_func(xx3)
                r = rearrange(r, 'b (h w) c -> b (w h) c', h=h, w=w) 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_2)
                x = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 


            elif self.layer_id%4==2:
                h, w = patch_resolution
                xx2 = torch.flip(xx, [3])
                # xx2 = torch.cat([xx2,self.hdconv(xx2)], dim=1)
                # xx2 = self.conv1x1(self.conv3x3(xx2))  
                xx2 = xx2 + self.alpha[4]*self.hdconv(xx2)
                xx2 = rearrange(xx2, 'b c h w -> b (h w) c', h=h,w=w) 
                r, k, v, g, w1 = self.jit_func(xx2)
                x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_1)
                # x = torch.flip(x,[1])
                x = torch.flip(x.view(B, h, w, C).contiguous(), [2]).view(B, T, C)
            else:
                h, w = patch_resolution
                xx4 = torch.flip(xx, [2])
                # xx4 = torch.cat([xx4,self.vdconv(xx4)], dim=1)
                # xx4 = self.conv1x1(self.conv3x3(xx4))  
                xx4 = xx4 + self.alpha[4]*self.vdconv(xx4)
                xx4 = rearrange(xx4, 'b c h w -> b (h w) c', h=h,w=w)

                r, k, v, g, w1 = self.jit_func(xx4)
                r = rearrange(r, 'b (h w) c -> b (w h) c', h=h, w=w) 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_2)
                x = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
                x = rearrange(x, 'b (h w) c -> b c h w ', h=h, w=w) 
                x = torch.flip(x, [2])
                x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)
                # x = torch.flip(v.view(B, h, w, C).contiguous(), [2]).view(B, T, C)
                # x = rearrange(x, 'b (w h) c -> b (h w) c', h=h, w=w) 


            return self.jit_func_2(x,g)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
    # def forward(self, x, patch_resolution=None):
    #     def _inner_forward(x):
    #         B, T, C = x.size()
    #         self.device = x.device
            
    #         # shortcut = x
    #         r, k, v, w = self.jit_func(x, patch_resolution)
    #         x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            
    #         # input_conv = shortcut.reshape([B,patch_resolution[0],patch_resolution[1],C]).permute(0, 3, 1, 2).contiguous()
    #         # out_33 = self.conv1a3(self.conv33(self.conv1b3(input_conv)))
    #         # output = out_33.permute(0, 2, 3, 1).contiguous()
    #         # x = output.reshape([B,T,C])*self.conv_scale + x
            
    #         return self.jit_func_2(x)
    #     if self.with_cp and x.requires_grad:
    #         x = cp.checkpoint(_inner_forward, x)
    #     else:
    #         x = _inner_forward(x)
    #     return x

# class eca_layer(nn.Module):
#     """Constructs a ECA module.

#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, channel, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#         return y
class eca_layer(nn.Module):
    """Constructs a modified ECA module with avg pooling and max pooling concatenation.
    
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size for Conv1d
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Average Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Max Pooling
        self.conv = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # Convolution to process concatenated features
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        avg_y = self.avg_pool(x)  # Average pooling
        max_y = self.max_pool(x)  # Max pooling

        # Squeeze and rearrange dimensions for Conv1d
        avg_y = avg_y.squeeze(-1).transpose(-1, -2)  # Shape (B, C, 1) -> (B, 1, C)
        max_y = max_y.squeeze(-1).transpose(-1, -2)  # Shape (B, C, 1) -> (B, 1, C)

        # Concatenate the results of avg_pool and max_pool along the channel dimension
        y = torch.cat([avg_y, max_y], dim=1)  # Shape (B, 2, C)

        # Apply Conv1d to the concatenated features
        y = self.conv(y)  # Shape (B, 1, C) -> (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # Back to (B, C, 1)

        # Apply sigmoid to generate attention weights
        y = self.sigmoid(y)
        return y


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False, 
                 with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        # self.head_size = self.attn_sz // self.n_head
        # print("VRWKV_ChannelMix head_size:", self.head_size)
        # assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        # self.shift_pixel = shift_pixel
        # self.shift_mode = shift_mode
        # self.shift_func = eval(shift_mode)

        hidden_rate = 3
        # self.multiconv = MultiConvBlock(in_channels=self.n_embd, out_channels=self.n_embd, theta=0.7)
        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        # self.value1 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.value2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1, bias=False,
		# 			  groups=n_embd),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.SiLU(),
		# )
        self.value = nn.Linear(hidden_sz,n_embd, bias=False)
        # self.receptance.init_scale = 0
        # self.value.init_scale = 0

        # self.conv1x1 = nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, groups=n_embd, bias=False)
        
        self.lpconv = Conv2d_lp(in_channels=self.n_embd, out_channels=self.n_embd, stride=1, padding=1, groups=1, bias=False)
        # self.adconv = Conv2d_ad(n_embd, n_embd, 3, bias=False,groups=1, theta=0.7)
        self.deconv = Conv2d_cd(n_embd, n_embd, 3, bias=False,groups=1, theta=0.7)
        self.rdconv = Conv2d_rd(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=False, theta=0.7)
        # # self.cdconv = ChannelDiffConv(n_embd, n_embd, bias=False,theta=0.7)
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        
        # self.conv1x1 = nn.Conv2d(in_channels=int(n_embd/4), out_channels=int(n_embd/4), kernel_size=1, groups=int(n_embd/4), bias=False)
        # self.deconv = Conv2d_cd(int(n_embd/4), int(n_embd/4), 3, bias=False,groups=1, theta=0.7)
        # self.rdconv = Conv2d_rd(in_channels=int(n_embd/4), out_channels=int(n_embd/4), kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=False, theta=0.7)
        # self.dpnet = DPNet(out_channels=n_embd, in_channels=n_embd, kernel_size=3, stride=1,dilation=1, groups=1, bias=False, theta=0.7)
        # self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        # self.omni_shift = OmniShift(dim=n_embd)
        # self.conv_scale = nn.Parameter(torch.ones(n_embd))

        # self.conv1b3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv1a3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.ca = eca_layer(n_embd)
        # self.conv33 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1, bias=False,
		# 			  groups=n_embd),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.SiLU(),
		# )
        # self.dwconv = nn.Sequential(nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1, bias=False,groups = n_embd),
        #                             nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1, bias=False,groups = 1),
        # )   

        # self.dw2conv = nn.Sequential(
        #     nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=2, padding=1, dilation=2, bias=False, groups=n_embd),
        #     nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
        # )
        # hidden_dim = max(n_embd // 4, 4)
        # self.channel_mlp = nn.Sequential(
        #     nn.Conv1d(5, 5, kernel_size=3,padding=1, groups=5, bias=False),  # 多尺度 depthwise
        #     nn.GELU(),
        #     nn.Conv1d(5, 1, kernel_size=1),  # 融合
        #     nn.Sigmoid()
        # )
        # self.dwconv = nn.Conv2d(
        #     n_embd, n_embd, kernel_size=3, 
        #     padding=1, groups=n_embd, bias=False
        # )
        # self.spatial_conv = nn.Conv2d(n_embd, 1, kernel_size=1)
        # self.spatial_act = nn.Sigmoid()
        # self.channel_mlp_refine = nn.Sequential(
        #     nn.Conv1d(2, 2, kernel_size=3,padding=1, groups=2, bias=False),  # 多尺度 depthwise
        #     nn.GELU(),
        #     nn.Conv1d(2, 1, kernel_size=1),  # 融合
        #     nn.Sigmoid()
        # )
        # self.dwconv_refine = nn.Conv2d(
        #     n_embd, n_embd, kernel_size=3,
        #     padding=1, groups=n_embd, bias=False
        # )
        # self.spatial_conv_refine = nn.Conv2d(n_embd, 1, kernel_size=1)
        # self.spatial_act = nn.Sigmoid()
        
    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        # def _inner_forward(x):
        #     # xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
        #     #                      with_cls_token=self.with_cls_token)
        #     # xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        #     # xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        #     B, T, C = x.size()
        #     shortcut = x
        #     input_conv = shortcut.reshape([B,patch_resolution[0],patch_resolution[1],C]).permute(0, 3, 1, 2).contiguous()
        #     out_33 = self.conv1a3(self.conv33(self.conv1b3(input_conv)))
        #     output = out_33.permute(0, 2, 3, 1).contiguous()
        #     x = output.reshape([B,T,C])*self.conv_scale + x

        #     xr, xk = x, x

        #     k = self.key(xk)
        #     k = torch.square(torch.relu(k))
        #     # if self.key_norm is not None:
        #     #     k = self.key_norm(k)
        #     kv = self.value(k)
        #     x = torch.sigmoid(self.receptance(xr)) * kv
        #     return x
        # if self.with_cp and x.requires_grad:
        #     x = cp.checkpoint(_inner_forward, x)
        # else:
        #     x = _inner_forward(x)
        # return x
        def _inner_forward(x):

            h, w = patch_resolution
            xx = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            # B, C = x.size(0), x.size(2)
            # low2 = F.interpolate(F.avg_pool2d(xx, 2), size=(h, w), mode='nearest')
            # low2_low = F.interpolate(F.avg_pool2d(low2, 2), size=(h, w), mode='nearest')
            # high2 = low2-low2_low
            # out21 = self.alpha[1] * xx - (1 - self.alpha[1]) * low2
            # out22 = self.alpha[0] * xx - (1 - self.alpha[0]) * high2
            
            # low4 = F.interpolate(F.avg_pool2d(xx, 4), size=(h, w), mode='nearest')
            # low4_low = F.interpolate(F.avg_pool2d(low4, 2), size=(h, w), mode='nearest')
            # high4 = low4 - low4_low
            # out41 = self.alpha[2] * xx - (1 - self.alpha[2]) * low4
            # out42 = self.alpha[0] * xx - (1 - self.alpha[0]) * high4
            # # low8 = F.interpolate(F.avg_pool2d(xx, 8), size=(h, w), mode='nearest')
            # # high8 = self.alpha[0] * xx - (1 - self.alpha[0]) * low4

            # gap_x = F.adaptive_avg_pool2d(xx, 1).view(B, 1, C)
            # gap_high2 = F.adaptive_avg_pool2d(out21, 1).view(B, 1, C)
            # gap_high4 = F.adaptive_avg_pool2d(out22, 1).view(B, 1, C)
            # gap_high6 = F.adaptive_avg_pool2d(out41, 1).view(B, 1, C)
            # gap_high8 = F.adaptive_avg_pool2d(out42, 1).view(B, 1, C)
            # gap_cat = torch.cat([gap_x, gap_high2, gap_high4, gap_high6, gap_high8], dim=1)
            # weights = self.channel_mlp(gap_cat).view(B, C, 1, 1)

            # w_s1 = self.spatial_act(self.spatial_conv(self.dwconv(xx)))
            # out1 = xx * weights + xx * w_s1
            
            # # gap_out1 = F.adaptive_avg_pool2d(out1, 1).view(B, 1, C)
            # # gap_refine = torch.cat([gap_x, gap_out1], dim=1)   # 用原始和一阶的结果一起 refine
            # # weights2 = self.channel_mlp_refine(gap_refine).view(B, C, 1, 1)

            # # w_s2 = self.spatial_act(self.spatial_conv_refine(self.dwconv_refine(out1)))
            # # out2 = out1 * weights2 + out1 * w_s2
            # out = rearrange(out1, 'b c h w -> b (h w) c', h=h, w=w)
            # return out
            de = self.deconv(xx)  # [B, C, H, W]
            # ad = self.adconv(xx)  # [B, C, H, W]
            rd = self.rdconv(xx)  # [B, C, H, W]
            lp = self.lpconv(xx)  # [B, C, H, W]
            # out1x1= self.conv1x1(xx)  # [B, C, H, W]
            # cd = self.cdconv(xx) # [B, C, H, W]
            xx = self.alpha[0]*xx + self.alpha[1]*lp + self.alpha[2]*de + self.alpha[3]*rd
            # xx,alpha = self.multiconv(xx)


            # # xx = self.omni_shift(xx)
            # xx = self.conv33(self.conv1b3(xx))
            # attention = self.dwconv(xx)  # [B, C, H, W]
            # attention = rearrange(attention, 'b c h w -> b (h w) c', h=h, w=w)  # [B, T, C]
            # attention = self.ca(xx)
            # xx1,xx2,xx3,xx4 = xx.chunk(4, dim=1)  # [B, C/4, H, W] * 4
            # de = self.deconv(xx4)  # [B, C, H, W]
            # rd = self.rdconv(xx3)  # [B, C, H, W]
            # out1x1= self.conv1x1(xx2)  # [B, C, H, W]

            # xx = torch.cat([xx1, out1x1, de, rd], dim=1)
            # xx = self.dpnet(xx)  # [B, C, H, W]
            xx = rearrange(xx, 'b c h w -> b (h w) c')   
            # xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
            #                      with_cls_token=self.with_cls_token)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            # xk = rearrange(xk, 'b (h w) c -> b c h w', h=h, w=w)
            # attention = self.dwconv(xk)  # [B, C, H, W]
            # attention = rearrange(attention, 'b c h w -> b (h w) c', h=h, w=w)  # [B, T, C]
            # xk = rearrange(xk, 'b (h w) c -> b c h w', h=h, w=w)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))

            # if self.key_norm is not None:
            #     k = self.key_norm(k)
            kv = self.value(k)
            # x =  kv * attention
            # x = rearrange(x, 'b c h w -> b (h w) c')  # [B, T, C]
            x = torch.sigmoid(xr) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, embed, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(n_embd,embed, n_head, n_layer, layer_id, shift_mode,
                                       shift_pixel, init_mode, key_norm=key_norm,
                                       with_cls_token=with_cls_token)

        self.ffn = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                    with_cls_token=with_cls_token)
        # self.ffn = CAB(n_embd)
        
        self.post_norm = post_norm
        
        self.gamma1 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.with_cp = with_cp
        self.relu = nn.ReLU(True)
    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            H, W = patch_resolution
            B, N, C = x.size()
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                x = self.gamma1*x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                x = self.gamma2*x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))       
            else:
                x = self.gamma1*x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                x = self.gamma2*x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)