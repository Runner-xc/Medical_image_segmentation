"""
评价指标

1.Recall 召回率=TP/(TP+FN)
2.Precision 精准率=TP/(TP+FP)
3.F1_score=2/(1/R+1/P)  # 召回率和准确率的调和平均数
4.Contour Loss
5.Boundary Loss

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Evaluate_Metric(nn.Module):
    def __init__(self, smooth=1e-5):
        super(Evaluate_Metric, self).__init__()
        self.class_names = [ 
                            'Aorta', 
                            'Gallbladder', 
                            'Spleen',
                            'Left Kidney',
                            'Right Kidney',
                            'Liver',
                            'Pancreas',
                            'Stomach']
        self.smooth = smooth
        
    def compute_confusion_matrix(self, img_pred, img_mask, threshold=0.5):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, 1, h, w) -> one_hot (batch, 4, h, w)
        """

        # 计算混淆矩阵的元素
        TP = torch.where((img_pred == 1) & (img_mask == 1), 1, 0).sum(dim=(-2, -1))
        TN = torch.where((img_pred == 0) & (img_mask == 0), 1, 0).sum(dim=(-2, -1))
        FP = torch.where((img_pred == 1) & (img_mask == 0), 1, 0).sum(dim=(-2, -1))
        FN = torch.where((img_pred == 0) & (img_mask == 1), 1, 0).sum(dim=(-2, -1))

        return TP, FN, FP, TN

    def recall(self, img_pred, img_mask):
        """"
        img_pred: 预测值 (batch, 3, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        # recall_dict = {}
        # class_names = self.class_names

        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=9).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=9).permute(0, 3, 1, 2).float() 

        # 计算总体召回率
        TP, FN, _, _ = self.compute_confusion_matrix(img_pred, img_mask)
        recall = TP / (TP + FN + self.smooth)
        recall = recall.mean(dim=0)
        
        Aorta            = recall[1].item()
        Gallbladder      = recall[2].item()
        Spleen           = recall[3].item()
        Left_Kidney      = recall[4].item()
        Right_Kidney     = recall[5].item()
        Liver            = recall[6].item()
        Pancreas         = recall[7].item()
        Stomach          = recall[8].item()
        recall           = recall[1:].mean().item()
        
        return Aorta, Gallbladder, Spleen, Left_Kidney, Right_Kidney, Liver, Pancreas, Stomach, recall

    def precision(self, img_pred, img_mask, threshold=0.5):
        # precision_dict = {}
        # class_names = self.class_names

        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=9).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=9).permute(0, 3, 1, 2).float() 
        
        # 计算总体精准率
        TP, _, FP, _ = self.compute_confusion_matrix(img_pred, img_mask)
        precision = TP / (TP + FP + self.smooth)
        precision = precision.mean(dim=0)
        
        Aorta = precision[1].item()
        Gallbladder = precision[2].item()
        Spleen = precision[3].item()
        Left_Kidney = precision[4].item()
        Right_Kidney = precision[5].item()
        Liver = precision[6].item()
        Pancreas = precision[7].item()
        Stomach = precision[8].item()
        # 计算总体精准率
        precision = precision[1:].mean()
        precision = precision.item()
        
        return Aorta, Gallbladder, Spleen, Left_Kidney, Right_Kidney, Liver, Pancreas, Stomach, precision


    def f1_score(self, img_pred, img_mask): 
        """
        recall:     召回率
        precision:  精准率
        """               
        Aorta_rc, Gallbladder_rc, Spleen_rc, Left_Kidney_rc, Right_Kidney_rc, Liver_rc, Pancreas_rc, Stomach_rc, recall = self.recall(img_pred, img_mask)
        Aorta_pr, Gallbladder_pr, Spleen_pr, Left_Kidney_pr, Right_Kidney_pr, Liver_pr, Pancreas_pr, Stomach_pr, precision= self.precision(img_pred, img_mask)

        # Aorta_F1
        Aorta_F1 = 2 * (Aorta_rc * Aorta_pr) / (Aorta_rc + Aorta_pr + self.smooth)
        # Gallbladder_F1
        Gallbladder_F1 = 2 * (Gallbladder_rc * Gallbladder_pr) / (Gallbladder_rc + Gallbladder_pr + self.smooth)
        # Spleen_F1
        Spleen_F1 = 2 * (Spleen_rc * Spleen_pr) / (Spleen_rc + Spleen_pr + self.smooth)
        # Left_Kidney_F1
        Left_Kidney_F1 = 2 * (Left_Kidney_rc * Left_Kidney_pr) / (Left_Kidney_rc + Left_Kidney_pr + self.smooth)
        # Right_Kidney_F1
        Right_Kidney_F1 = 2 * (Right_Kidney_rc * Right_Kidney_pr) / (Right_Kidney_rc + Right_Kidney_pr + self.smooth)
        # Liver_F1
        Liver_F1 = 2 * (Liver_rc * Liver_pr) / (Liver_rc + Liver_pr + self.smooth)
        # Pancreas_F1
        Pancreas_F1 = 2 * (Pancreas_rc * Pancreas_pr) / (Pancreas_rc + Pancreas_pr + self.smooth)
        # Stomach_F1
        Stomach_F1 = 2 * (Stomach_rc * Stomach_pr) / (Stomach_rc + Stomach_pr + self.smooth)
        # 计算总体F1_score
        F1_score = 2 * (recall * precision) / (recall + precision + self.smooth)
        
        return Aorta_F1, Gallbladder_F1, Spleen_F1, Left_Kidney_F1, Right_Kidney_F1, Liver_F1, Pancreas_F1, Stomach_F1, F1_score
    

    def dice_coefficient(self, logits, targets):
        """
        dice 指数
        """
        num_classes = logits.shape[1]
        # 预处理
        logits = torch.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 计算总体dice
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        dice = (2 * intersection) / (union + self.smooth)
        
        Aorta = dice[1].item()
        Gallbladder = dice[2].item()
        Spleen = dice[3].item()
        Left_Kidney = dice[4].item()
        Right_Kidney = dice[5].item()
        Liver = dice[6].item()
        Pancreas = dice[7].item()
        Stomach = dice[8].item()
        # 计算总体dice
        dice = dice[1:].mean().item()

        return Aorta, Gallbladder, Spleen, Left_Kidney, Right_Kidney, Liver, Pancreas, Stomach, dice

    def mIoU(self, logits, targets):
        """
        mIoU: 平均交并比
        """
        num_classes = logits.shape[1]
        logits = torch.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=num_classes).permute(0, 3, 1, 2).float()
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 计算总体mIoU
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1)) - intersection
        iou =  intersection / (union + self.smooth)
        
        Aorta = iou[1].item()
        Gallbladder = iou[2].item()
        Spleen = iou[3].item()
        Left_Kidney = iou[4].item()
        Right_Kidney = iou[5].item()
        Liver = iou[6].item()
        Pancreas = iou[7].item()
        Stomach = iou[8].item()
        # 计算总体mIoU
        mIoU = iou[1:].mean().item()
        
        return Aorta, Gallbladder, Spleen, Left_Kidney, Right_Kidney, Liver, Pancreas, Stomach, mIoU
    
    def accuracy(self, logits, targets):
        """
        accuracy: 准确率
        """
        # 预处理
        img_pred = torch.argmax(logits, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=9).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(targets, num_classes=9).permute(0, 3, 1, 2).float() 
        
        # 计算总体精准率
        TP, FN, FP, TN = self.compute_confusion_matrix(img_pred, img_mask)
        
        accuracy = (TP + TN) / (TP + TN + FN + FP + self.smooth)
        accuracy = accuracy.mean(dim=0)
        Aorta = accuracy[1].item()
        Gallbladder = accuracy[2].item()
        Spleen = accuracy[3].item()
        Left_Kidney = accuracy[4].item()
        Right_Kidney = accuracy[5].item()
        Liver = accuracy[6].item()
        Pancreas = accuracy[7].item()
        Stomach = accuracy[8].item()
        # 计算总体精准率
        accuracy = accuracy[1:].mean().item()
        
        return Aorta, Gallbladder, Spleen, Left_Kidney, Right_Kidney, Liver, Pancreas, Stomach, accuracy

    def update(self, img_pred, img_mask):
        """
        更新评价指标
        """
        recall = self.recall(img_pred, img_mask)
        precision = self.precision(img_pred, img_mask)
        dice = self.dice_coefficient(img_pred, img_mask)
        f1_score = self.f1_score(img_pred, img_mask)
        mIoU = self.mIoU(img_pred, img_mask)
        accuracy = self.accuracy(img_pred, img_mask)
        
        metrics = [recall, precision, dice, f1_score, mIoU, accuracy]
        metrics = np.stack(metrics, axis=0)
        metrics = np.nan_to_num(metrics)

        return metrics