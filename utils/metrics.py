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


class Metrics(nn.Module):
    def __init__(self, class_names:list, smooth=1e-5):
        super(Metrics, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.smooth = smooth
        
    def compute_confusion_matrix(self, prediction, target, threshold=0.5):
        """
        prediction: 预测值 (batch, 4, h, w)
        target:     标签值 (batch, 1, h, w) -> one_hot (batch, 4, h, w)
        """

        # 计算混淆矩阵的元素
        TP = torch.where((prediction == 1) & (target == 1), 1, 0).sum(dim=(-2, -1))
        TN = torch.where((prediction == 0) & (target == 0), 1, 0).sum(dim=(-2, -1))
        FP = torch.where((prediction == 1) & (target == 0), 1, 0).sum(dim=(-2, -1))
        FN = torch.where((prediction == 0) & (target == 1), 1, 0).sum(dim=(-2, -1))

        return TP, FN, FP, TN

    def recall(self, prediction, target):
        """"
        prediction: 预测值 (batch, 3, h, w)
        target: 标签值 (batch, h, w)
        """
        # 预处理
        prediction = torch.argmax(prediction, dim=1).to(dtype=torch.int64)
        prediction = F.one_hot(prediction, num_classes=9).permute(0, 3, 1, 2).float() 
        target = F.one_hot(target, num_classes=9).permute(0, 3, 1, 2).float() 

        # 计算总体召回率
        TP, FN, _, _ = self.compute_confusion_matrix(prediction, target)
        recall = TP / (TP + FN + self.smooth)
        recall = recall.mean(dim=0)
        recalls = [recall[i+1].item() for i in range(self.num_classes)]
        recalls.append(recall[1:].mean().item())
        return recalls

    def precision(self, prediction, target, threshold=0.5):
        # 预处理
        prediction = torch.argmax(prediction, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        prediction = F.one_hot(prediction, num_classes=9).permute(0, 3, 1, 2).float() 
        target = F.one_hot(target, num_classes=9).permute(0, 3, 1, 2).float() 
        
        # 计算总体精准率
        TP, _, FP, _ = self.compute_confusion_matrix(prediction, target)
        precision = TP / (TP + FP + self.smooth)
        precision = precision.mean(dim=0)
        precisions = [precision[i+1].item() for i in range(self.num_classes)]
        precisions.append(precision[1:].mean().item())
        return precisions


    def f1_score(self, prediction, target): 
        """
        recall:     召回率
        precision:  精准率
        """               
        recalls = self.recall(prediction, target)
        precisions = self.precision(prediction, target)
        f1_scores = [2 * (recalls[i] * precisions[i]) / (recalls[i] + precisions[i] + self.smooth) for i in range(self.num_classes + 1)]
        return f1_scores
    

    def dice_coefficient(self, logits, targets):
        """
        dice 指数
        """
        # 预处理
        logits = torch.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=self.num_classes+1).permute(0, 3, 1, 2).float()
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=self.num_classes+1).permute(0, 3, 1, 2).float()
        
        # 计算总体dice
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        dice = (2 * intersection) / (union + self.smooth)
        dices = [dice[i+1].item() for i in range(self.num_classes)]
        dices.append(dice[1:].mean().item())
        return dices

    def mIoU(self, logits, targets):
        """
        mIoU: 平均交并比
        """
        logits = torch.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=self.num_classes+1).permute(0, 3, 1, 2).float()
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=self.num_classes+1).permute(0, 3, 1, 2).float()
        
        # 计算总体mIoU
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1)) - intersection
        iou =  intersection / (union + self.smooth)
        mIoUs = [iou[i+1].item() for i in range(self.num_classes)]
        mIoUs.append(iou[1:].mean().item())
        return mIoUs
    
    def accuracy(self, logits, targets):
        """
        accuracy: 准确率
        """
        # 预处理
        prediction = torch.argmax(logits, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        prediction = F.one_hot(prediction, num_classes=self.num_classes+1).permute(0, 3, 1, 2).float() 
        target = F.one_hot(targets, num_classes=self.num_classes+1).permute(0, 3, 1, 2).float() 
        
        # 计算总体精准率
        TP, FN, FP, TN = self.compute_confusion_matrix(prediction, target)
        accuracy = (TP + TN) / (TP + TN + FN + FP + self.smooth)
        accuracy = accuracy.mean(dim=0)
        accuracies = [accuracy[i+1].item() for i in range(self.num_classes)]
        accuracies.append(accuracy[1:].mean().item())
        return accuracies

    def update(self, prediction, target):
        """
        更新评价指标
        """
        recalls = self.recall(prediction, target)
        precisions = self.precision(prediction, target)
        dices = self.dice_coefficient(prediction, target)
        f1_scores = self.f1_score(prediction, target)
        mIoUs = self.mIoU(prediction, target)
        accuracys = self.accuracy(prediction, target)
        
        metrics = [recalls, precisions, dices, f1_scores, mIoUs, accuracys]
        metrics = np.stack(metrics, axis=0)
        metrics = np.nan_to_num(metrics)

        return metrics