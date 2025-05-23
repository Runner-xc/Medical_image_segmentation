import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from mamba_ssm import Mamba


"""-------------------------------------------------Convolution----------------------------------------------"""
import torch
from torchvision.ops import DeformConv2d

class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义偏移量生成网络（普通卷积层）
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * 3 * 3,  # 每个位置x/y偏移量 × 3x3卷积核采样点数
            kernel_size=3, 
            padding=1
        )
        
        # 可变形卷积层
        self.deform_conv = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1
        )
        self._initialize_weights()
    def forward(self, x):
        # 生成偏移量 [B, 18, H, W]
        offsets = self.offset_conv(x)
        # 执行可变形卷积
        return self.deform_conv(x, offsets)
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(DWConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, dilation=dilation)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        return self.pconv(self.dconv(x))
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels // 4
        # 初始定义分离的各层
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(mid_channels), 
            self.relu)
        
        self.cbr2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            self.relu)
        self._initialize_weights()
        
    def forward(self, x):
        x = self.cbr1(x)  
        return self.cbr2(x)

    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class Axis_wise_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(Axis_wise_Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel), padding=(0, kernel // 2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(kernel, 1), padding=(kernel // 2, 0), groups=in_channels))
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        return self.pwconv(self.conv(x))

    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class AWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(AWConv, self).__init__()
        self.conv = nn.Sequential(
            Axis_wise_Conv2d(in_channels, out_channels, kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self._initialize_weights()
        
    def forward(self, x):
        return self.conv(x)
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class Att_AWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(Att_AWConv, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            Axis_wise_Conv2d(in_channels, out_channels, kernel),
            nn.BatchNorm2d(out_channels),
            self.relu)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            self.relu,
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1))
        self.sgm = nn.Sigmoid()
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv(x)
        weight = self.sgm(self.fc(self.gap(x)))
        return x * weight
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class Att_AWBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.awconv5x5 = AWConv(in_channels, out_channels, kernel=5)
        self.se = SE_Block(out_channels)
        self._initialize_weights()
    def forward(self, x):
        return self.se(self.awconv5x5(x))
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class DWDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels // 4
        # 调整顺序为 DWConv → BN → ReLU
        self.relu = nn.ReLU(inplace=True)
        
        self.dconv1 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            self.relu
        )
        self.dconv2 = nn.Sequential(
            DWConv(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.relu
        )   
        self._initialize_weights()

    def forward(self, x):
        return self.dconv2(self.dconv1(x))
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class Conv_3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Conv_3, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 4
        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))

        self.c2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(mid_channels), 
                                nn.ReLU(inplace=True))

        self.c3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(out_channels), 
                                nn.ReLU(inplace=True))
        self._initialize_weights()

    def forward(self, x1):
        return self.c3(self.c2(self.cbr1(x1)))
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Dalit_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Dalit_Conv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3), 
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
        self._initialize_weights()

    def forward(self, x):
        return self.cbr3(self.cbr2(self.cbr1(x)))
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class DWDalit_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DWDalit_Conv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        self.cbr1 = nn.Sequential(DWConv(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(DWConv(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(DWConv(mid_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3), 
                                nn.BatchNorm2d(out_channels), 
                                nn.ReLU(inplace=True))
        self._initialize_weights()

    def forward(self, x):
        return self.cbr3(self.cbr2(self.cbr1(x)))
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConv, self).__init__()

        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), 
                                nn.BatchNorm2d(out_channels)) 
              
        self.cbr2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(out_channels), 
                                  nn.ReLU(inplace=True))
        
        self.c3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(out_channels))
        self._initialize_weights()        
    def forward(self, x):
        residual = x
        re = self.c1(residual)
        x = self.cbr2(x)
        x = self.c3(x)
        x = x + re
        return self.relu(x)
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class ResDConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResDConv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3), 
                                  nn.BatchNorm2d(out_channels))
        self._initialize_weights()
    def forward(self, x):
        residual = x
        re = self.c1(residual)
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = x + re
        return self.relu(x)
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class DWResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWResConv, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), 
                               nn.BatchNorm2d(out_channels))
        
        self.cbr2 = nn.Sequential(DWConv(in_channels, out_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(out_channels), 
                                  nn.ReLU(inplace=True))
        self.cbr3 = nn.Sequential(DWConv(out_channels, out_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(out_channels))
        self._initialize_weights()      
    def forward(self, x):
        residual = x
        re = self.cbr1(residual)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = x + re
        return self.relu(x)
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DWResDConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DWResDConv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(DWConv(in_channels, mid_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(DWConv(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(DWConv(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3), 
                                  nn.BatchNorm2d(out_channels))
        self._initialize_weights()
    def forward(self, x):
        residual = x
        re = self.c1(residual)
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = x + re
        return self.relu(x)
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,      # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width 1×1
                expand=expand,    # Block expansion factor
        )
        self._initialize_weights()
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out
    
    def _initialize_weights(self):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
"""---------------------------------------------Pyramid Pooling Module----------------------------------------------------"""
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播，直接返回大于0的输入作为1，否则为0
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播，使用hardtanh函数来处理梯度
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        return STEFunction.apply(x)
    
# 因果图块（Causality Map Block）
class CausalityMapBlock(nn.Module):
    def __init__(self):
        super(CausalityMapBlock, self).__init__()
        # print("CausalityMapBlock initialized")

    def forward(self, x):
        """
        计算因果图，输入x的形状为(bs, k, n, n)，其中bs是批量大小，k是特征图的数量，
        n是特征图的空间维度。
        
        :param x: 网络的潜在特征
        :return: 因果图，形状为(bs, k, k)
        """
        if torch.isnan(x).any():
            # print("...the current feature maps object contains NaN")
            raise ValueError

        # 计算每个特征图的最大值
        maximum_values = torch.max(torch.flatten(x, 2), dim=2)[0]
        MAX_F = torch.max(maximum_values, dim=1)[0]
        x_div_max = x / (MAX_F.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-8)

        x = torch.nan_to_num(x_div_max, nan=0.0)

        # 计算因果图
        sum_values = torch.sum(torch.flatten(x, 2), dim=2)
        sum_values = torch.nan_to_num(sum_values, nan=0.0)
        maximum_values = torch.max(torch.flatten(x, 2), dim=2)[0]
        mtrx = torch.einsum('bi,bj->bij', maximum_values, maximum_values)
        tmp = mtrx / (sum_values.unsqueeze(1) + 1e-8)
        causality_maps = torch.nan_to_num(tmp, nan=0.0)

        # 归一化因果图
        max_cmaps = torch.max(causality_maps, dim=1, keepdim=True)[0]
        min_cmaps = torch.min(causality_maps, dim=1, keepdim=True)[0]
        causality_maps = (causality_maps - min_cmaps) / (max_cmaps - min_cmaps + 1e-8)

        # print(f"causality_maps: min {torch.min(causality_maps)}, max {torch.max(causality_maps)}, mean {torch.mean(causality_maps)}")
        return causality_maps


# 因果因子提取器（Causality Factors Extractor）
class CausalityFactorsExtractor(nn.Module):
    def __init__(self):
        super(CausalityFactorsExtractor, self).__init__()
        self.STE = StraightThroughEstimator()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # print("CausalityFactorsExtractor initialized")

    def forward(self, x, causality_maps):
        """
        根据因果图计算因果因子，输入x的形状为(bs, k, h, w)，其中bs是批量大小，k是特征图的数量，
        h和w是特征图的空间维度。因果图的形状为(bs, k, k)。
        
        :param x: 网络的潜在特征
        :param causality_maps: 因果图
        :return: 因果加权的特征图
        """
        triu = torch.triu(causality_maps, 1)
        tril = torch.tril(causality_maps, -1).permute((0, 2, 1)).contiguous()

        e = tril - triu
        e = self.STE(e)
        e = e.permute((0, 2, 1))

        f = triu - tril
        f = self.STE(f)
        bool_matrix = e + f

        by_col = torch.sum(bool_matrix, 2)
        by_row = torch.sum(bool_matrix, 1)

        multiplicative_factors = by_col - by_row
        multiplicative_factors = self.relu(multiplicative_factors)
        max_factors = torch.max(multiplicative_factors, dim=1, keepdim=True)[0]
        min_factors = torch.min(multiplicative_factors, dim=1, keepdim=True)[0]
        multiplicative_factors = (multiplicative_factors - min_factors) / (max_factors - min_factors + 1e-8)

        # print(f"multiplicative_factors: min {torch.min(multiplicative_factors)}, max {torch.max(multiplicative_factors)}, mean {torch.mean(multiplicative_factors)}")
        return torch.einsum('bkmn,bk->bkmn', x, multiplicative_factors)
class PSPModule(nn.Module):
    def __init__(self, in_channels: int, bin_size_list: list = [1, 2, 4, 8]):
        super(PSPModule, self).__init__()
        branch_channels = in_channels // 4        ## C/4
        self.branches = nn.ModuleList()
        # CAB 模块
        self.causality_map_block = CausalityMapBlock()
        self.causality_factors_extractor = CausalityFactorsExtractor()
        for i in range(len(bin_size_list)):
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin_size_list[i]),  # 使用平均池化
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU()
            )
            self.branches.append(branch)       

    def forward(self, inputs):
        if not self.branches:
            return inputs
        final = None
        for i, branch in enumerate(self.branches):
            out = branch(inputs)
            # # CAB 模块
            # causality_maps = self.causality_map_block(out)
            # enhanced_features = self.causality_factors_extractor(out, causality_maps)
            
            out = F.interpolate(out, size=inputs.shape[2:], mode='bilinear', align_corners=True)
            if final is None:
                final = out
            else:
                final = torch.cat([final, out], dim=1)
        final = torch.cat([inputs, final], dim=1)  # 将各特征图在通道维上拼接起来
        return final

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features
class DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DenseASPPBlock, self).__init__()
        self.aspp_1 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 1, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_2 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 2, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_3 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 3, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_5 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 5, 0.1,
                                      norm_layer, norm_kwargs)
        # self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                    #   norm_layer, norm_kwargs)
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 4, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        aspp1 = self.aspp_1(x)
        x = torch.cat([aspp1, x], dim=1)

        aspp2 = self.aspp_2(x)
        x = torch.cat([aspp2, x], dim=1)

        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp5 = self.aspp_5(x)
        x = torch.cat([aspp5, x], dim=1)

        # aspp24 = self.aspp_24(x)
        # x = torch.cat([aspp24, x], dim=1)

        x = self.cbr1(x)
        return x
    
"""---------------------------------------------AMSFN----------------------------------------------------""" 
class AMSFN(nn.Module):  #Adaptive Convolutional Pooling Network (ACPN)
    def __init__(self, in_channels, out_channels=None, mid_channels=None,):
        super(AMSFN, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.conv1x1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))

        # 3×3
        self.cbr1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbr2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbr3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sgm = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self._initialize_weights()

    def forward(self, x):
        b, c, h, w = x.size()
        c1 = self.conv1x1(x)
        s1 = self.avg_pool(c1)

        c2 = self.cbr1(x)
        s2 = self.avg_pool(c2)

        c3 = self.cbr2(x)
        s3 = self.avg_pool(c3)

        c4 = self.cbr3(x)
        s4 = self.avg_pool(c4)

        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.mlp(out)
        weights = self.sgm(out)

        # 将权重作用到各个卷积结果上并相加
        c1_weighted = c1 * weights[:, 0:c   , ...]
        c2_weighted = c2 * weights[:, c:c*2 , ...]
        c3_weighted = c3 * weights[:, c*2:c*3, ...]
        c4_weighted = c4 * weights[:, c*3:  , ...]

        output = c1_weighted + c2_weighted + c3_weighted + c4_weighted
        output = self.final_conv(output)
        return output
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
class AMSFNV2(AMSFN):  #Adaptive Convolutional Pooling Network (ACPN)
    def __init__(self, in_channels, out_channels=None, mid_channels=None,):
        super(AMSFNV2, self).__init__(in_channels, 
                                      out_channels, 
                                      mid_channels)
        if out_channels is None:
            out_channels = in_channels
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.conv1x1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))

        # 3×3
        self.cbr1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbr2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbr3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return super(AMSFNV2, self).forward(x)

class SpatialChannelAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        if in_ch % 4 != 0:
            out_ch = 1
        else:
            out_ch = in_ch // 4
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch, in_ch, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_ch, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self._initialize_weights()  
        
    def forward(self, x):
        channel_weight = self.channel_att(x)  # [B,C,1,1]
        spatial_weight = self.spatial_att(x)  # [B,1,H,W]
        return x * channel_weight * spatial_weight  # 双重注意力
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class AMSFNV3(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        # 分支1: 1x1卷积保留细节
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            SpatialChannelAttention(in_channels)
        )
        
        # 分支2: 轴向卷积
        self.branch2 = AWConv(in_channels, in_channels, kernel=5)
        
        # 分支3: 可变形卷积适应形状
        self.branch3 = nn.Sequential(
            DeformConvBlock(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 动态融合权重生成
        self.fusion_weights = nn.Conv2d(in_channels*3, 3, 3, padding=1)  # 空间敏感权重
        
        # 输出变换
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x):
        # 多流并行计算 提高训练速度
        s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        
        with torch.cuda.stream(s1):
            b1 = self.branch1(x)
        with torch.cuda.stream(s2):
            b2 = self.branch2(x)
        with torch.cuda.stream(s3):
            b3 = self.branch3(x)
        torch.cuda.synchronize()

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # 生成空间注意力权重
        weight_map = torch.softmax(self.fusion_weights(torch.cat([b1, b2, b3], dim=1)), dim=1)
        w1, w2, w3 = weight_map.chunk(3, dim=1)
        
        # 加权融合
        fused = w1*b1 + w2*b2 + w3*b3
        return self.final(fused)
    
    def _initialize_weights(self, init_gain=0.02):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class AMSFNV4(AMSFNV3):
    def __init__(self, in_channels, out_channels=None):
        super().__init__(
            in_channels, out_channels
        )
        if out_channels is None:
            out_channels = in_channels
        
        # 分支2: 轴向卷积
        self.branch2 = Att_AWConv(in_channels, in_channels, kernel=5)
        
        # 动态融合权重生成
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(in_channels*3,3,3, padding=1, groups=3),
            nn.Conv2d(3,3,1)
        )
        
        # 输出变换
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        # 多流并行计算 提高训练速度
        s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        
        with torch.cuda.stream(s1):
            b1 = self.branch1(x)
        with torch.cuda.stream(s2):
            b2 = self.branch2(x)
        with torch.cuda.stream(s3):
            b3 = self.branch3(x)
        torch.cuda.synchronize()

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # 生成空间注意力权重
        weight_map = torch.softmax(self.fusion_weights(torch.cat([b1, b2, b3], dim=1)), dim=1)
        w1, w2, w3 = weight_map.chunk(3, dim=1)
        
        # 加权融合
        fused = w1*b1 + w2*b2 + w3*b3
        return self.final(fused)

if __name__ == '__main__':
    from attention import *
    x = torch.randn(16, 3, 256, 256).to('cuda')
    model = MambaLayer(dim=3, d_state=64, d_conv=4).to('cuda')
    out = model(x)
    print(out.shape, "\n",
          model)
    
elif os.path.dirname(os.path.abspath(__file__)) == '/mnt/e/VScode/WS-Hub/Linux-md_seg/Medical_image_segmentation/model/utils':
    from .attention import *

else:
    from model.utils.attention import *

        
     

        