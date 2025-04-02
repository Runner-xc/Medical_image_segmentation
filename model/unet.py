"""
unet
"""
import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter 
from .utils.model_info import calculate_computation  

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)

        self.down1 = Down(base_channels, base_channels*2)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0)

        self.down2 = Down(base_channels*2, base_channels*4)
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)

        self.down3 = Down(base_channels*4, base_channels*8)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0) 

        self.up1 = Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)

        self.up2 = Up(base_channels * 4 , base_channels * 2, bilinear=bilinear)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)

        self.up3 = Up(base_channels * 2 , base_channels,     bilinear=bilinear)

        self.up4 = Up(base_channels,      base_channels,     bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)             # [1, 64, 320, 320]
        x1 = self.encoder_dropout1(x1)

        x2 = self.down1(x1)             # [1, 128, 160, 160]
        x2 = self.encoder_dropout2(x2)

        x3 = self.down2(x2)             # [1, 256, 80, 80]
        x3 = self.encoder_dropout3(x3)

        x4 = self.down3(x3)             # [1, 512, 40, 40]
        x4 = self.encoder_dropout4(x4)

        x5 = self.pool(x4)              # [1, 512, 20, 20]           
        x = self.center_conv(x5)        # [1, 512, 20, 20]
        x = self.bottleneck_dropout(x)
           
        x = self.up1(x, x4)             # [1, 256, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up2(x, x3)             # [1, 128, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up3(x, x2)             # [1, 64, 160, 160]

        x = self.up4(x, x1)             # [1, 64, 320, 320]
        logits = self.out_conv(x)       # [1, c, 320, 320]       
        return logits 
          
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class ResD_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(ResD_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inconv = DoubleConv(in_channels, base_channels)

        self.down1 = ResD_Down(base_channels, base_channels*2)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0)

        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.8 if p!=0 else 0)

        self.down3 = ResD_Down(base_channels*4, base_channels*8)
        self.encoder_dropout4 = nn.Dropout2d(p=p)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=min(p+0.1, 0.7) if p!=0.0 else 0.0) 

        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.5 if p!=0 else 0)

        self.up2 = ResD_Up(base_channels * 4 , base_channels * 2, bilinear=bilinear)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)

        self.up3 = ResD_Up(base_channels * 2 , base_channels,     bilinear=bilinear)
        self.decoder_dropout3 = nn.Dropout2d(p=p*0.1 if p!=0 else 0)

        self.up4 = Up(base_channels,      base_channels,     bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)             # [1, 64, 320, 320]

        x2 = self.down1(x1)             # [1, 128, 160, 160]
        x2 = self.encoder_dropout2(x2)

        x3 = self.down2(x2)             # [1, 256, 80, 80]
        x3 = self.encoder_dropout3(x3)

        x4 = self.down3(x3)             # [1, 512, 40, 40]
        x4 = self.encoder_dropout4(x4)

        x5 = self.pool(x4)              # [1, 512, 20, 20]           
        x = self.center_conv(x5)        # [1, 512, 20, 20]
        x = self.bottleneck_dropout(x)
           
        x = self.up1(x, x4)             # [1, 256, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up2(x, x3)             # [1, 128, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up3(x, x2)             # [1, 64, 160, 160]
        x = self.decoder_dropout3(x)

        x = self.up4(x, x1)             # [1, 64, 320, 320]
        logits = self.out_conv(x)       # [1, c, 320, 320]       
        return logits
    
            
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
        
