import torch
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter 
from model.utils.model_info import calculate_computation

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
    
from model.aicunet import AICUNet

class DC_UNet(AICUNet):
    def __init__(self,
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 K=4,
                 temperature=34,
                 ):
        super(DC_UNet, self).__init__(in_channels, n_classes, p, base_channels)
        self.encoder1 = Dynamic_conv2d(in_channels,       base_channels,   kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        self.encoder2 = Dynamic_conv2d(base_channels,     base_channels*2, kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        self.encoder3 = Dynamic_conv2d(base_channels*2,   base_channels*4, kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        self.encoder4 = Dynamic_conv2d(base_channels*4,   base_channels*8, kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)

        self.decoder1 = Dynamic_conv2d(base_channels*16,  base_channels*4, kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        self.decoder2 = Dynamic_conv2d(base_channels*8,   base_channels*2, kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        self.decoder3 = Dynamic_conv2d(base_channels*4,   base_channels,   kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        self.decoder4 = Dynamic_conv2d(base_channels*2,   base_channels,   kernel_size=3, ratio=0.25, padding=1, K=K, temperature=temperature)
        
    def forward(self, x):
        return super(DC_UNet, self).forward(x)

if __name__ == '__main__':
    model = DC_UNet(in_channels=3, n_classes=4, p=0)
    summary(model,(1, 3, 256, 256))
    # ==========================================================================================
    # Total params: 7,181,596
    # Trainable params: 7,181,596
    # Non-trainable params: 0
    # Total mult-adds (Units.MEGABYTES): 612.92
    # ==========================================================================================
    # Input size (MB): 0.79
    # Forward/backward pass size (MB): 5.25
    # Params size (MB): 9.83
    # Estimated Total Size (MB): 15.86
    # ==========================================================================================
    calculate_computation(model, input_size=(3, 256, 256))
    # ========================================
    # Input size: (3, 256, 256)
    # FLOPs: 2.48 GFLOPs
    # MACs: 1.24 GMACs
    # Params: 4.82 M
    # ========================================