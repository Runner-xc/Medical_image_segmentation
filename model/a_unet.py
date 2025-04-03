import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter
from model.utils.attention import *
from model.utils.modules import * 
 
class A_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(A_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 编码器
        self.encoder1 = DoubleConv(in_channels,       base_channels, )
        self.encoder2 = DoubleConv(base_channels,     base_channels*2)
        self.encoder3 = DoubleConv(base_channels*2,   base_channels*4)
        self.encoder4 = DoubleConv(base_channels*4,   base_channels*8)
        # 多尺度融合
        self.acpn0 = AMSFN(base_channels)
        self.acpn1 = AMSFN(base_channels*2) 
        self.acpn2 = AMSFN(base_channels*4) 
        self.acpn3 = AMSFN(base_channels*8)
        # encoder_dropout
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0) 
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)
                                    # 编码器更高dropout
        # bottleneck
        ## self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0)

        # 解码器
        self.decoder1 = DoubleConv(base_channels * 16 ,   base_channels * 4)
        self.decoder2 = DoubleConv(base_channels * 8,     base_channels * 2)
        self.decoder3 = DoubleConv(base_channels * 4,     base_channels)
        self.decoder4 = DoubleConv(base_channels * 2,     base_channels)
        # 多尺度融合
        self.acpn5 = AMSFN(base_channels*4)
        self.acpn6 = AMSFN(base_channels*2) 
        self.acpn7 = AMSFN(base_channels) 
        self.acpn8 = AMSFN(base_channels)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)

        # 输出层
        self.out_conv = DoubleConv(base_channels, n_classes)

    def forward(self, x):
        e1 = self.encoder1(x)
        x1 = self.acpn0(e1)                                 
        x2 = self.down(x1)
        x2 = self.encoder_dropout1(x2)                                

        e2 = self.encoder2(x2)                              # [b, 64, 128, 128]
        a2 = self.acpn1(e2)
        x3 = self.down(a2)
        x3 = self.encoder_dropout2(x3)
        
        e3 = self.encoder3(x3)                              # [b, 128, 64, 64]
        a3 = self.acpn2(e3)
        x4 = self.down(a3)
        x4 = self.encoder_dropout3(x4)

        e4 = self.encoder4(x4)                              # [b, 256, 32, 32]
        a4 = self.acpn3(e4)
        x5 = self.down(a4)
        x5 = self.encoder_dropout4(x5)
                            
        c5 = self.center_conv(x5)                           # [b, 512, 16, 16]
        c5 = self.bottleneck_dropout(c5)

        d4 = self.up(c5)
        d4 = torch.cat([d4, e4], dim=1)                     # [b, 256, 32, 32]
        d4 = self.decoder1(d4)
        d4 = self.acpn5(d4)
        x = self.decoder_dropout1(d4)   

        d3 = self.up(x)
        d3 = torch.cat([d3, e3], dim=1)                      # [b, 128, 64, 64]
        d3 = self.decoder2(d3)
        d3 = self.acpn6(d3)
        x = self.decoder_dropout2(d3)

        d2 = self.up(x)
        d2 = torch.cat([d2, e2], dim=1)                      # [b, 64, 128, 128]
        d2 = self.decoder3(d2)
        d2 = self.acpn7(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)                     # [b, 64, 256, 256]
        d1 = self.decoder4(d1)
        x = self.acpn8(d1)
        logits = self.out_conv(x)                           # [1, c, 256, 256]
        
        return logits
    
