"""
损失函数:Dice Loss、BoundaryLoss、Focal Loss、DistanceBasedLoss、Tversky Loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from monai.losses import DiceLoss, HausdorffDTLoss

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


"""
Cross Entropy Loss
"""
class CrossEntropyLoss():
    def __init__(self, class_names):
        """
        class_names:['Aorta', 
                     'Gallbladder', 
                     'Spleen',
                     'Left Kidney',
                     'Right Kidney',
                     'Liver',
                     'Pancreas']          
        """
        self.class_names = class_names

    def __call__(self, logits, targets):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        
        # 计算loss
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        targets = targets[:, 1:, ...]
        ce = nn.CrossEntropyLoss(label_smoothing=1e-7, reduction='none', ignore_index=0)    # 不进行缩减会返回（batch, h, w）的loss值
       
        # celoss期望的logits是(b, c, h, w), targets是(b, h, w)
        loss_dict = {}
        for i in range(num_classes):
            target = targets[:, i, ...]
            logit = logits[:, i, ...] 
            loss_dict[self.class_names[i]] = ce(logit, target)   # [b, 256, 256]  

        # 记录类别损失
        total_loss = torch.stack(list(loss_dict.values())).mean()
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
class diceloss():
    def __init__(self, class_names, smooth=1e-5):
        """
        smooth: 平滑值
        """
        self.smooth = smooth
        self.class_names = class_names

    def __call__(self, logits, targets):
        """
        img_pred: 预测值 (batch, 8, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        tensor_one = torch.tensor(1)

        # logits argmax 
        logits = torch.softmax(logits, dim=1)  # 不能直接使用argmax，会丢失grad_fn,因为argmax不可导
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  
        logits = logits[:, 1:, ...]
        targets = targets[:, 1:, ...]
        # 计算总的损失
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        dice = (2 * intersection) / (union + self.smooth)
        loss = tensor_one - dice
        total_loss = loss.sum() / num_classes
        
        # 计算每个类别的损失
        loss_dict = {name: loss[i] for i, name in enumerate(self.class_names)}
        loss_dict['total_loss'] = total_loss
        return loss_dict

class IOULoss():
    def __init__(self, class_names, smooth=1e-5):
        """
        smooth: 平滑值
        """
        self.smooth = smooth
        self.class_names = class_names

    def __call__(self, logits, targets):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        tensor_one = torch.tensor(1)

        # logits argmax 
        logits = torch.softmax(logits, dim=1)  # 不能直接使用argmax，会丢失grad_fn,因为argmax不可导
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  
        logits = logits[:, 1:, ...]
        targets = targets[:, 1:, ...]
        # 计算总的损失
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        iou =  intersection / (union - intersection + self.smooth)
        loss = tensor_one - iou
        total_loss = loss.sum() / num_classes
        
        # 计算每个类别的损失
        loss_dict = {}
        loss_dict = {name: loss[i] for i, name in enumerate(self.class_names)}
        loss_dict['total_loss'] = total_loss
        return loss_dict

class AdaptiveSegLoss(nn.Module):
    def __init__(self, num_classes, init_alpha=0.5):
        """
        num_classes: 类别数(包含背景)
        """
        super().__init__()
        # 每个类别独立的可学习参数
        self.alphas = nn.Parameter(torch.ones(num_classes-1)*init_alpha, 
                                  requires_grad=True)
        
        # 初始化基础损失函数
        self.dice_loss = DiceLoss(include_background=False,  
                                reduction='none')  # 关闭默认reduction
        self.hd_loss = HausdorffDTLoss(include_background=False, 
                                     reduction='none')  # 关闭默认reduction
        
        # # 可学习的缩放参数
        # self.beta = nn.Parameter(torch.tensor(0.001), requires_grad=True)
        
        self.names = ['Organic matter', 'Organic pores', 'Inorganic pores']
        assert len(self.names) == num_classes-1, "类别名数量不匹配"

    def forward(self, y_pred, y_true):
        """
        y_pred: [B, C, H, W] 未经 softmax
        y_true: [B, 1, H, W] 或 [B, H, W] (值为0~C-1的整数)
        """
        # 预处理
        B, C, H, W = y_pred.shape
        y_pred = F.softmax(y_pred, dim=1)  # [B,C,H,W]
        y_true_oh = F.one_hot(y_true.squeeze(1), num_classes=C).permute(0,3,1,2).float()  # [B,C,H,W]
        
        # 计算各通道损失（排除背景）
        dice = self.dice_loss(y_pred, y_true_oh)
        dice = dice.flatten(start_dim=2).squeeze(-1)  # [B,C-1]
        
        hd = self.hd_loss(y_pred, y_true_oh)
        hd = hd.flatten(start_dim=2).squeeze(-1)  # [B,C-1]
        # scaled_hd = self.beta * hd
        
        # 动态权重计算
        alphas = torch.sigmoid(self.alphas).to(y_pred.device)  # [C-1,]
        
        # 逐类别加权
        class_losses = alphas * dice.mean(dim=0) + (1 - alphas) * hd.mean(dim=0)
        
        # 汇总损失
        loss_dict = {
            'total_loss': class_losses.mean(),
            **{name: loss for name, loss in zip(self.names, class_losses)}
        }
        
        return loss_dict
        
class Focal_Loss():
    """
    γ : 聚焦因子,用于控制损失的敏感度
    α : 平衡正负样本权重
    """
    def __init__(self, class_names, alpha=0.25, gamma=2):
        self.gamma=gamma
        self.alpha=alpha
        self.class_names = class_names

    def __call__(self,logits, targets):
        """
        img_pred: 预测值 (batch, 3, h, w)
        img_mask: 标签值 (batch, h, w)
        """   
        num_classes = logits.shape[1]
        
        # 计算loss
        targets = targets.to(torch.int64)
        ce = nn.CrossEntropyLoss(label_smoothing=1e-7, reduction='none', ignore_index=0)    # 不进行缩减会返回（batch, h, w）的loss值
        
        # 计算 Focal Loss
        # celoss期望的logits是(b, c, h, w), targets是(b, h, w)
        ce_loss = ce(logits, targets)  # [b, 256, 256]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        # 计算每个类别的损失，忽略背景（索引0）
        loss_dict = {}
        total_loss = 0
        for i in range(1, num_classes):  # 从1开始，忽略背景
            class_loss = focal_loss[targets == i].mean()
            loss_dict[self.class_names[i-1]] = class_loss
            total_loss += class_loss    

        # 计算总损失
        total_loss /= num_classes-1  # 减去背景类
        loss_dict['total_loss'] = total_loss
        return loss_dict
class WDiceLoss():
    
    def __init__(self, class_names, smooth=1e-5):
        """
        smooth: 平滑值
        """
        self.smooth = smooth
        self.class_names = class_names

    def __call__(self, logits, targets, weights=[0.1,0.2,0.7]):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        tensor_one = torch.tensor(1)
        # logits argmax 
        preds = torch.softmax(logits, dim=1)

        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()     
        # 计算每个类别的损失
        loss_dict = {}
        total_loss = 0.0
        for i in range(1,num_classes):
            pred = preds[:, i, ...]
            target = targets[:, i, ...]
            # 计算总的损失
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2 * intersection) / (union + self.smooth)
            loss = tensor_one - dice
            # 加权
            loss = loss * weights[i-1]
            loss_dict[self.class_names[i-1]] = loss 
            total_loss += loss 
        loss_dict['total_loss'] = total_loss
        return loss_dict

"""
DWDLoss 动态加权loss
""" 
class DWDLoss(nn.Module):
    def __init__(self, class_names, smooth=1e-8):
        self.class_names = class_names
        self.smooth = smooth
    def calculate_cnum(self, targets):
        cnum = []
        for i in range(1,4):
            cnum.append(torch.sum(targets==i))
        return torch.tensor(cnum)
    
    def calculate_weights(self, n, c):
        max_n = torch.max(n)
        return torch.log(max_n / n[c-1]) + 1
    
    def __call__(self, logits, targets):
        num_classes = logits.shape[1]
        # 处理logits
        preds = torch.softmax(logits, dim=1)
        targets = targets.to(torch.int64)
        masks = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # 类别权重
        total_loss = 0.0
        loss_dict = {f"{name}": 0.0 for name in self.class_names}
        one = torch.tensor(1).to(targets.device)

        # 动态加权loss
        for i in range(1,num_classes):
            # 计算权重
            cnum = self.calculate_cnum(targets).to(targets.device)
            weight = self.calculate_weights(cnum, i).to(targets.device)

            # 计算dice + loss
            pred = preds[:, i, ...]    # 预测
            mask = masks[:, i, ...]    # 标签
            intersection = (pred * mask).sum()
            union = (pred.sum() + mask.sum())
            ip = intersection / masks[:, i, ...].sum()
            dice = (2 * intersection) / (union + self.smooth)
            # single_loss
            w_dice = weight ** (1-ip) * dice 
            loss = one - w_dice
            loss_dict[self.class_names[i-1]] = loss
            # add
            total_loss += loss
     
        # 计算总的损失
        loss_dict['total_loss'] = total_loss
        return loss_dict  

            

# if __name__ == '__main__':
#     loss_fn = TotalLoss(flag=True, loss_fn='boundary_loss')
#     x = torch.randn(1, 4, 256, 256)
#     y = torch.randint(0, 4, (1, 256, 256))
#     x = F.softmax(x, dim=1)
#     dice_loss = DiceLoss()
#     boundary_loss = SurfaceLoss()
#     focal_loss = Focal_Loss()
#     distance_based_loss = DistanceBasedLoss()
#     print(focal_loss(x, y))
#     print(boundary_loss(x, y))
#     print(dice_loss(x, y))
