"""
unet
"""
import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter 
from .utils.model_info import calculate_computation  
  
class UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.dowm = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels*2)
        self.encoder3 = DoubleConv(base_channels*2, base_channels*4)
        self.encoder4 = DoubleConv(base_channels*4, base_channels*8)
        # encoder_dropout
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0)
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)
        # bottleneck
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0)
        # 解码器
        self.decoder1 = DoubleConv(base_channels * 16 , base_channels * 4) 
        self.decoder2 = DoubleConv(base_channels * 8 ,  base_channels * 2)
        self.decoder3 = DoubleConv(base_channels * 4 ,  base_channels)
        self.decoder4 = DoubleConv(base_channels * 2,   base_channels)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.encoder1(x)               # [1, 32, 320, 320]
        x2 = self.dowm(x1)
        x2 = self.encoder_dropout1(x2)

        x2 = self.encoder2(x2)              # [1, 64, 160, 160]
        x3 = self.dowm(x2)
        x3 = self.encoder_dropout2(x3)

        x3 = self.encoder3(x3)              # [1, 128, 80, 80]
        x4 = self.dowm(x3)
        x4 = self.encoder_dropout3(x4)

        x4 = self.encoder4(x4)              # [1, 256, 40, 40]
        x5 = self.dowm(x4)
        x5 = self.encoder_dropout4(x5)
       
        x = self.center_conv(x5)            # [1, 256, 20, 20]
        x = self.bottleneck_dropout(x)
        
        x = self.up(x)                      # [1, 256, 40, 40]
        x = torch.cat([x, x4], dim=1)       # [1, 256+256, 40, 40]
        x = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up(x)                      # [1, 128, 80, 80]
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        x = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up(x)                      # [1, 64, 160, 160]
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        x = self.decoder3(x)                # [1, 32, 160, 160]

        x = self.up(x)                      # [1, 32, 160, 160]
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        
        logits = self.out_conv(x)           # [1, c, 320, 320]       
        return logits 
          
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class ResD_UNet(UNet):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(ResD_UNet, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            p=p,
            base_channels=base_channels)
        # 编码器
        self.encoder2 = ResDConv(base_channels,   base_channels*2)
        self.encoder3 = ResDConv(base_channels*2, base_channels*4) 
        self.encoder4 = ResDConv(base_channels*4, base_channels*8)  

        # 解码器
        self.decoder1 = ResDConv(base_channels * 8 , base_channels * 4)   
        self.decoder2 = ResDConv(base_channels * 4,  base_channels * 2)
        self.decoder3 = ResDConv(base_channels * 2,  base_channels)
        
    def forward(self, x):      
        return super(ResD_UNet, self).forward(x)
    
            
if __name__ == '__main__':
    from utils.attention import EMA
    from utils.modules import *   
    model = UNet(in_channels=3, n_classes=4, p=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(3,256,256)
    summary(model, (1, 3, 256, 256))
    calculate_computation(model, input_size=(3, 256, 256), device=device)
    # ========================================
    # Input size: (3, 256, 256)
    # FLOPs: 28.87 GFLOPs
    # MACs: 14.44 GMACs
    # Params: 9.04 M
    # ========================================

else:
    from model.utils.attention import *
    from model.utils.modules import * 
        
