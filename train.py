import torch
import datetime
from torch.utils.data import DataLoader, Dataset
from utils.medicine_data import Synapse_data
from utils import data_split
from utils.writing_logs import writing_logs
import argparse
import os
from torch.optim import Adam, SGD, RMSprop, AdamW
import time
from model.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large
from model.pspnet import PSPNet
from model.Segnet import SegNet
from model.u2net import u2net_full_config, u2net_lite_config
from model.unet import UNet, ResD_UNet
from model.aicunet import AICUNet
from model.a_unet import A_UNet
from model.m_unet import M_UNet
from model.rdam_unet import RDAM_UNet, DWRDAM_UNet
from model.vm_unet import VMUNet
from model.dc_unet import DC_UNet
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from utils import param_modification
from utils import write_experiment_log
from utils.loss_fn import *
from utils.metrics import Evaluate_Metric
from torch.utils.tensorboard import SummaryWriter
import utils.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Union, List
from utils.run_tensorboard import run_tensorboard
from model.Segnet import SegNet
from torchinfo import summary
import swanlab

# 预处理
class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485), std=(0.229)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(base_size),
            # T.RandomCrop(crop_size),
            # T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data

class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485), std=(0.229)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(base_size),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data  
    
# ===================== 表情符号配置 =====================
PARAM_ICONS = {
    'model': '📦', 'lr': '📚', 'wd': '⚖️', 'dropout': '☔',
    'l1_lambda': 'λ¹', 'l2_lambda': 'λ²', 'scheduler': '🔄',
    'loss_fn': '💥', 'best_epoch': '🏆', 'time': '🕒', 'cost': '⏳'
}   

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

detailed_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")

def format_epoch_header(epoch, end_epoch):
                return f"\n🚀✨========= Epoch {epoch+1}/{end_epoch} =========✨🚀\n"

def build_params_block(args):
    return (
        f"{PARAM_ICONS['model']} model : {args.model}\n"
        f"{PARAM_ICONS['lr']} lr : {args.lr}\n"
        f"{PARAM_ICONS['wd']} wd : {args.wd}\n"
        f"{PARAM_ICONS['dropout']} dropout : {args.dropout_p}\n"
        f"{PARAM_ICONS['l1_lambda']} l1_lambda : {args.l1_lambda}\n"
        f"{PARAM_ICONS['l2_lambda']} l2_lambda : {args.l2_lambda}\n"
        f"{PARAM_ICONS['loss_fn']} loss_fn : {args.loss_fn}\n"
        f"{PARAM_ICONS['scheduler']} scheduler : {args.scheduler}\n"        
                )

def main(args, aug_args):
    """——————————————————————————————————————————————打印初始配置———————————————————————————————————————————————"""
    # 将args转换为字典
    params = vars(args)

    # 映射参数名称到参数
    param_map = {
        'lr'            : '1. lr',
        'wd'            : '2. wd',
        'l1_lambda'     : '3. l1_lambda',
        'l2_lambda'     : '4. l2_lambda',
        'elnloss'       : '5. elnloss',
        'dropout_p'     : '6. dropout_p',
        'model'         : '7. model',
        'loss_fn'       : '8. loss_fn',
        'optimizer'     : '9. optimizer',
        'scheduler'     : '10. scheduler',
        'Tmax'          : '11. Tmax',
        'eta_min'       : '12. eta_min',
        'save_flag'     : '13. save_flag',
        'batch_size'    : '14. batch_size',
        'num_small_data': '15. num_small_data',
        'eval_interval' : '16. eval_interval',
        'split_flag'    : '17. split_flag',
        'resume'        : '18. resume'
    }

    # 筛选需要打印的参数
    printed_params = list(param_map.keys())
    params_dict = {}
    params_dict['Parameter'] = [param_map[p] for p in printed_params]
    params_dict['Value'] = [str(params[p]) for p in printed_params if p in params]

    # 打印参数
    params_header = ['Parameter', 'Value']
    # print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    
    """——————————————————————————————————————————————记录修改配置———————————————————————————————————————————————"""
    initial_time = time.time()
    if args.change_params:    
        x = input("是否需要修改配置参数：\n 0. 不修改, 继续。 \n\
请输入需要修改的参数序号（int）： ")
        
        args = param_modification.param_modification(args, x)
    save_modification_path = f"{args.results_path}/{args.modification}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}"

    """——————————————————————————————————————————————加载数据集——————————————————————————————————————————————"""
    # 定义设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio   

    # 划分数据集
    if args.num_small_data is not None:
        train_datasets, val_datasets, test_datasets = data_split.small_data_split_to_train_val_test(args.data_path, 
                                                                                   num_small_data = args.num_small_data, 
                                                                                   # train_ratio=0.8, 
                                                                                   # val_ratio=0.1, 
                                                                                   save_root_path = args.data_root_path,
                                                                                   flag           = args.split_flag) 
    
    else:
        train_datasets, val_datasets, test_datasets = data_split.data_split_to_train_val_test(aug_args, args.data_path, train_ratio=train_ratio, val_ratio=val_ratio,
                            save_root_path=args.data_root_path,   # 保存划分好的数据集路径
                            flag=args.split_flag)

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio   

    # 读取数据集
    train_datasets = Synapse_data(train_datasets, 
                            transforms=SODPresetTrain((256, 256), crop_size=256))
    
    val_datasets = Synapse_data(val_datasets, 
                            transforms=SODPresetEval((256, 256)))
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataloader = DataLoader(train_datasets, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    val_dataloader = DataLoader(val_datasets, 
                                batch_size=8, 
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=True)
    """——————————————————————————————————————————————模型 配置———————————————————————————————————————————————"""   
    # 加载模型
    model_map = {

            # UNet 系列
            "u2net_full"                    : u2net_full_config(),
            "u2net_lite"                    : u2net_lite_config(),
            "unet"                          : UNet(in_channels=1, n_classes=9, base_channels=32, bilinear=True, p=args.dropout_p),
            "ResD_unet"                     : ResD_UNet(in_channels=1, n_classes=9, base_channels=32, bilinear=True, p=args.dropout_p),
            "a_unet"                        : A_UNet(in_channels=1, n_classes=9, base_channels=32, bilinear=True, p=args.dropout_p),
            "m_unet"                        : M_UNet(in_channels=1, n_classes=9, base_channels=32, bilinear=True, p=args.dropout_p),
            "rdam_unet"                     : RDAM_UNet(in_channels=1, n_classes=9, base_channels=32, bilinear=True, p=args.dropout_p),
            "dwrdam_unet"                   : DWRDAM_UNet(in_channels=1, n_classes=9, base_channels=32, bilinear=True, p=args.dropout_p),
            "aicunet"                       : AICUNet(in_channels=1, n_classes=9, base_channels=32, p=args.dropout_p),
            "vm_unet"                       : VMUNet(input_channels=1, num_classes=9),
            "dc_unet"                       : DC_UNet(in_channels=1, n_classes=9, p=args.dropout_p),

            # 其他架构
            "Segnet"                        : SegNet(n_classes=9, dropout_p=args.dropout_p),
            "pspnet"                        : PSPNet(classes=9, dropout=args.dropout_p, pretrained=False),
            "deeplabv3_resnet50"            : deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=9),
            "deeplabv3_resnet101"           : deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=9),
            "deeplabv3_mobilenetv3_large"   : deeplabv3_mobilenetv3_large(aux=False, pretrain_backbone=False, num_classes=9)
        }
    model = model_map.get(args.model)
    if not model:
        raise ValueError(f"Invalid model name: {args.model}")
    
    # 初始化模型
    kaiming_initial(model)
    model.to(device)
    model_info = str(summary(model, (1, 1, 256, 256)))  
    
    """——————————————————————————————————————————————优化器 调度器——————————————————————————————————————————————"""
    # 优化器 
    optim_map = {
            'AdamW' : lambda: AdamW(model.parameters(), 
                                    args.lr, 
                                    weight_decay=args.wd,
                                    betas=(0.95, 0.999),
                                    eps=1e-8
                                    ),

            'SGD'   : lambda:   SGD(model.parameters(), 
                                    args.lr, 
                                    momentum=0.9, 
                                    weight_decay=args.wd),
                               
          'RMSprop' : lambda:RMSprop(model.parameters(), 
                                    args.lr, 
                                    alpha=0.9, 
                                    eps=1e-8, 
                                    weight_decay=args.wd)
        }
    optimizer = optim_map.get(args.optimizer, optim_map['AdamW'])()  
        
    # 调度器
    if args.scheduler == 'CosineAnnealingLR':
        # 计算总batch数和warmup步数
        num_batches_per_epoch = len(train_dataloader)
        warmup_steps = args.warmup_epochs * num_batches_per_epoch  # 总预热步数
        Tmax_steps = args.Tmax * num_batches_per_epoch  # 将Tmax从epoch转换为step
        
        # 获取初始学习率
        lr_initial = optimizer.param_groups[0]['lr']

        # 定义带Warmup的Lambda调度器
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                # Warmup阶段：线性增长
                (step / warmup_steps) if step < warmup_steps
                # 正常阶段：余弦退火
                else (args.eta_min + (lr_initial - args.eta_min) * 
                    (1 + math.cos(math.pi * (step - warmup_steps) / Tmax_steps)) / 2) / lr_initial
            ),
            last_epoch=-1)  # 初始步数从0开始 
          
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='min', 
                                      factor=0.1, 
                                      patience=5, 
                                      threshold=1e-4, 
                                      threshold_mode='rel', 
                                      cooldown=0, 
                                      min_lr=0, 
                                      eps=1e-8)
    else:
        print(f"wrong scaler name{args.scheduler}")
        
    # 损失函数 
    loss_map = {
            'CrossEntropyLoss'  : CrossEntropyLoss(),
            'DiceLoss'          : diceloss(),
            'FocalLoss'         : Focal_Loss(),
            'WDiceLoss'         : WDiceLoss(),
            'DWDLoss'           : DWDLoss(),
            'IoULoss'           : IOULoss(),
            'dice_hd'           : AdaptiveSegLoss(4)
        }
    loss_fn = loss_map.get(args.loss_fn)
    
    # 缩放器
    scaler = torch.amp.GradScaler() if args.amp else None
    Metrics = Evaluate_Metric()
    
    # 日志保存路径
    save_logs_path = f"{args.results_path}/{args.log_name}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}"
    
    if not os.path.exists(save_logs_path):
        os.makedirs(save_logs_path)
    if args.save_flag:
        if args.elnloss:
            log_path = f'{save_logs_path}/optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}'
            writer = SummaryWriter(log_path)
        else:
            log_path = f'{save_logs_path}/optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}'
            writer = SummaryWriter(log_path)
        
        # 转换参数为字典并过滤需要记录的参数
        config = vars(args)
        excluded_params = ['data_path', 'data_root_path', 'save_scores','results_path', 
                         'save_weight', 'log_name', 'modification',
                         'device', 'resume', 'save_flag', 'split_flag', 'change_params']
        config = {k: v for k, v in config.items() if k not in excluded_params}
        
        # 初始化SwanLab（整个训练过程只初始化一次）
        swanlab.init(
            project="Medical_image_segmentation",
            experiment_name=f"{args.model}-{args.loss_fn}-{detailed_time_str}",
            config=config,
        )
    
    """——————————————————————————————————————————————参数 列表———————————————————————————————————————————————"""
    # 记录修改后的参数
    if args.elnloss:
        modification_log_name = f"optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}.md"
    else:
        modification_log_name = f"optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}.md"
    params = vars(args)
    params_dict['Parameter'] = printed_params
    params_dict['Value'] = [str(params[p]) for p in printed_params]
    contents = tabulate(params_dict, headers=params_header, tablefmt="grid")

    mdf = os.path.join(save_modification_path, modification_log_name)
    if not os.path.exists(os.path.dirname(mdf)):
        os.makedirs(os.path.dirname(mdf))
    if args.save_flag:
        write_experiment_log.write_exp_logs(mdf, contents) 
    
    """参数列表"""
    params = vars(args)
    params_dict = {}
    params_dict['Parameter'] = [str(p[0]) for p in list(params.items())]
    params_dict['Value'] = [str(p[1]) for p in list(params.items())]
    params_header = ['Parameter', 'Value']
    """打印参数"""
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
            
    """——————————————————————————————————————————————训练 验证——————————————————————————————————————————————"""
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
  
    best_mean_loss, current_miou = float('inf'), 0.0
    best_epoch = 0 
    patience = 0 
    current_mean_loss = float('inf')

    """断点续传"""    
    if args.resume:
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(args.resume, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler.last_epoch = checkpoint['step']
        best_mean_loss = checkpoint['best_mean_loss']
        start_epoch = checkpoint['best_epoch']
        best_epoch = checkpoint['best_epoch']
        print(f"🌐Resume from epoch: {start_epoch}")
    
    """训练"""   
    for epoch in range(start_epoch, end_epoch):
        
        print(f"\n ✈️»»——————————————««  Epoch {epoch+1}/{end_epoch}  »»——————————————««✈️\n")
        print(f"🌈 ---- Training ---- 🌈")
        # 记录时间
        start_time = time.time()
        # 训练
        # Aorta_loss, Gallbladder_loss, Left_Kidney_loss, Right_Kidney_loss, Liver_loss, Pancreas_loss, Spleen_loss, Stomach_loss, train_loss, Metric_list
        Aorta_loss, Gallbladder_loss, Left_Kidney_loss, Right_Kidney_loss, Liver_loss, Pancreas_loss, Spleen_loss, Stomach_loss, train_loss, T_Metric_list = train_one_epoch(model, 
                                                                                    optimizer, 
                                                                                    epoch, 
                                                                                    train_dataloader, 
                                                                                    device=device, 
                                                                                    loss_fn=loss_fn, 
                                                                                    scaler=scaler,
                                                                                    Metric=Metrics,
                                                                                    scheduler = scheduler,
                                                                                    elnloss=args.elnloss,     #  Elastic Net正则化
                                                                                    l1_lambda=args.l1_lambda,
                                                                                    l2_lambda=args.l2_lambda) # loss_fn=loss_fn, 
        
        # 求平均
        train_Aorta_loss   = Aorta_loss / len(train_dataloader)
        train_Gallbladder_loss = Gallbladder_loss / len(train_dataloader)
        train_Left_Kidney_loss = Left_Kidney_loss / len(train_dataloader)
        train_Right_Kidney_loss = Right_Kidney_loss / len(train_dataloader)
        train_Liver_loss   = Liver_loss / len(train_dataloader)
        train_Pancreas_loss = Pancreas_loss / len(train_dataloader)
        train_Spleen_loss  = Spleen_loss / len(train_dataloader)
        train_Stomach_loss = Stomach_loss / len(train_dataloader)
        train_loss       = train_loss / len(train_dataloader)
        
        train_loss_list = [train_Aorta_loss, 
                           train_Gallbladder_loss, 
                           train_Left_Kidney_loss, 
                           train_Right_Kidney_loss, 
                           train_Liver_loss, 
                           train_Pancreas_loss, 
                           train_Spleen_loss,
                           train_Stomach_loss, 
                           train_loss]
        
        # 评价指标 metrics = [recall, precision, dice, f1_score]
        train_metrics               ={}
        train_metrics["Loss"]       = train_loss_list
        train_metrics["Recall"]     = T_Metric_list[0]
        train_metrics["Precision"]  = T_Metric_list[1]
        train_metrics["Dice"]       = T_Metric_list[2]
        train_metrics["F1_scores"]  = T_Metric_list[3]
        train_metrics["mIoU"]       = T_Metric_list[4]
        train_metrics["Accuracy"]   = T_Metric_list[5]
        
        # 结束时间
        end_time = time.time()
        train_cost_time = end_time - start_time

        # 打印
        print(
            f"💧train_Aorta_loss: {train_Aorta_loss:.3f}\n"
            f"💧train_Gallbladder_loss: {train_Gallbladder_loss:.3f}\n"
            f"💧train_Left_Kidney_loss: {train_Left_Kidney_loss:.3f}\n"
            f"💧train_Right_Kidney_loss: {train_Right_Kidney_loss:.3f}\n"
            f"💧train_Liver_loss: {train_Liver_loss:.3f}\n"
            f"💧train_Pancreas_loss: {train_Pancreas_loss:.3f}\n"
            f"💧train_Spleen_loss: {train_Spleen_loss:.3f}\n"
            f"💧train_Stomach_loss: {train_Stomach_loss:.3f}\n"
            f"💧train_loss: {train_loss:.3f}\n"
            f"🕒train_cost_time: {train_cost_time/60:.2f}mins\n")
        

        """验证"""
        if epoch % args.eval_interval == 0 or epoch == end_epoch - 1:
            print(f"🌈 ---- Validation ---- 🌈")
            # 记录验证开始时间
            start_time = time.time()
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            v_Aorta_loss, v_Gallbladder_loss, v_Left_Kidney_loss, v_Right_Kidney_loss, v_Liver_loss, v_Pancreas_loss, v_Spleen_loss, v_Stomach_loss, mean_loss, Metric_list = evaluate(model, device, val_dataloader, loss_fn, Metrics) # val_loss, recall, precision, f1_scores

            # 求平均
            v_Aorta_loss     = v_Aorta_loss / len(val_dataloader)
            v_Gallbladder_loss = v_Gallbladder_loss / len(val_dataloader)
            v_Left_Kidney_loss = v_Left_Kidney_loss / len(val_dataloader)
            v_Right_Kidney_loss = v_Right_Kidney_loss / len(val_dataloader)
            v_Liver_loss     = v_Liver_loss / len(val_dataloader)
            v_Pancreas_loss  = v_Pancreas_loss / len(val_dataloader)
            v_Spleen_loss    = v_Spleen_loss / len(val_dataloader)
            v_Stomach_loss   = v_Stomach_loss / len(val_dataloader)
            val_mean_loss   = mean_loss / len(val_dataloader)
            # loss_list
            val_loss_list   = [v_Aorta_loss,
                               v_Gallbladder_loss, 
                               v_Left_Kidney_loss, 
                               v_Right_Kidney_loss, 
                               v_Liver_loss, 
                               v_Pancreas_loss, 
                               v_Spleen_loss,
                               v_Stomach_loss,
                               val_mean_loss]
            
            # 获取当前学习率
            current_lr = scheduler.get_last_lr()[0]  

            # 评价指标 metrics = [recall, precision, dice, f1_score]
            val_metrics                 ={}
            val_metrics["Loss"]         = val_loss_list
            val_metrics["Recall"]       = Metric_list[0]
            val_metrics["Precision"]    = Metric_list[1]
            val_metrics["Dice"]         = Metric_list[2]
            val_metrics["F1_scores"]    = Metric_list[3]
            val_metrics["mIoU"]         = Metric_list[4]
            val_metrics["Accuracy"]     = Metric_list[5]
            # 验证====结束时间
            end_time = time.time()
            val_cost_time = end_time - start_time

            # 打印结果
            print(
                f"🔥val_Aorta_loss: {v_Aorta_loss:.3f}\n"
                f"🔥val_Gallbladder_loss: {v_Gallbladder_loss:.3f}\n"
                f"🔥val_Left_Kidney_loss: {v_Left_Kidney_loss:.3f}\n"
                f"🔥val_Right_Kidney_loss: {v_Right_Kidney_loss:.3f}\n"
                f"🔥val_Liver_loss: {v_Liver_loss:.3f}\n"
                f"🔥val_Pancreas_loss: {v_Pancreas_loss:.3f}\n"
                f"🔥val_Spleen_loss: {v_Spleen_loss:.3f}\n"
                f"🔥val_Stomach_loss: {v_Stomach_loss:.3f}\n"
                f"🔥val_mean_loss: {val_mean_loss:.3f}\n"
                f"🕒val_cost_time: {val_cost_time:.2f}s")
            print(f"🚀Current learning rate: {current_lr:.7f}")
            
            # 记录日志
            tb = args.tb
            if tb:
                writing_logs(writer, train_metrics, val_metrics, epoch) 
                # 新增SwanLab日志记录
                swanlab.log({
                    # 训练指标
                    "train/Aorta_loss": train_Aorta_loss,
                    "train/Gallbladder_loss": train_Gallbladder_loss,
                    "train/Left_Kidney_loss": train_Left_Kidney_loss,
                    "train/Right_Kidney_loss": train_Right_Kidney_loss,
                    "train/Liver_loss": train_Liver_loss,
                    "train/Pancreas_loss": train_Pancreas_loss,
                    "train/Spleen_loss": train_Spleen_loss,
                    "train/Stomach_loss": train_Stomach_loss,
                    "train/train_loss": train_loss,
                    "train/recall": train_metrics["Recall"][-1],
                    "train/precision": train_metrics["Precision"][-1],
                    "train/dice": train_metrics["Dice"][-1],
                    "train/f1_scores": train_metrics["F1_scores"][-1],
                    "train/mIoU": train_metrics["mIoU"][-1],
                    "train/accuracy": train_metrics["Accuracy"][-1],
                    # 验证指标
                    "val/Aorta_loss": v_Aorta_loss,
                    "val/Gallbladder_loss": v_Gallbladder_loss,
                    "val/Left_Kidney_loss": v_Left_Kidney_loss,
                    "val/Right_Kidney_loss": v_Right_Kidney_loss,
                    "val/Liver_loss": v_Liver_loss,
                    "val/Pancreas_loss": v_Pancreas_loss,
                    "val/Spleen_loss": v_Spleen_loss,
                    "val/Stomach_loss": v_Stomach_loss,
                    "val/val_mean_loss": val_mean_loss,
                    "val/recall": val_metrics["Recall"][-1],
                    "val/precision": val_metrics["Precision"][-1],
                    "val/dice": val_metrics["Dice"][-1],
                    "val/f1_scores": val_metrics["F1_scores"][-1],
                    "val/mIoU": val_metrics["mIoU"][-1],
                    "val/accuracy": val_metrics["Accuracy"][-1],
                    
                    # 学习率
                    "learning_rate": current_lr,
                    
                    # 时间指标
                    "time/epoch_time": train_cost_time + val_cost_time
                }, step=epoch)              
                """-------------------------TXT--------------------------------------------------------"""        
                writer.add_text('val/Metrics', 
                                f"optim: {args.optimizer}, lr: {args.lr}, wd: {args.wd}, l1_lambda: {args.l1_lambda}, l2_lambda: {args.l2_lambda}"+ '\n'
                                f"model: {args.model}, loss_fn: {args.loss_fn}, scheduler: {args.scheduler}"
                                )
                if epoch == 5:
                    run_tensorboard(log_path)               
            
            # 保存指标
            if best_mean_loss >= val_mean_loss:
                best_mean_loss = val_mean_loss
                best_epoch = epoch + 1
        
            # ===================== 静态配置 =====================
            metrics_table_header = ['Metrics_Name', 'Mean', 'Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']  # 原始表头
            metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']        

            epoch_s = format_epoch_header(epoch, end_epoch)
            params_block = build_params_block(args)

            # 指标表格
            metrics_dict = {name: val_metrics[name] for name in metrics_table_left}
            metrics_table = [
                [name,  
                f"{metrics_dict[name][-1]:.5f}",  # 平均
                f"{metrics_dict[name][0]:.5f}",   # OM
                f"{metrics_dict[name][1]:.5f}",   # OP
                f"{metrics_dict[name][2]:.5f}"]   # IOP
                for name in metrics_table_left
            ]

            training_info = (
                f"{PARAM_ICONS['time']} time : {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}"
                f"\n🍎 Train Loss: {train_loss:.3f} "
                f"| 🍏 Val Loss: {val_mean_loss:.3f}\n"
                f"{PARAM_ICONS['best_epoch']} best_epoch : {best_epoch}\n"
                f"{PARAM_ICONS['cost']} val_cost_time : {val_cost_time/60:.2f} mins"
            )

            # 输出
            write_info = (
                epoch_s +
                "\n========= 训练配置 =========\n" +
                params_block +
                "\n========= 性能指标 =========\n" +
                tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid') + "\n" +
                "\n========= 训练状态 =========\n" +
                training_info
            )

            print(write_info)

            # 保存结果
            save_scores_path = f'{args.results_path}/{args.save_scores}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}'
            if args.elnloss:
                results_file = f"optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}.txt"
            else:
                results_file = f"optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}.txt"
            file_path = os.path.join(save_scores_path, results_file)

            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            if args.save_flag:
                with open(file_path, "a") as f:
                    f.write(write_info)                      
       
        if args.save_flag:
            # 保存best模型
            if args.elnloss:
                save_weights_path = f"{args.results_path}/{args.save_weight}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}/optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}"  # 保存权重路径
            else:
                save_weights_path = f"{args.results_path}/{args.save_weight}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}/optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}"
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)

            save_file = {"model"        : model.state_dict(),
                        "optimizer"     : optimizer.state_dict(),
                        "Metrics"       : Metrics.state_dict(),
                        "scheduler"     : scheduler.state_dict(),
                        "best_mean_loss": best_mean_loss,
                        "best_epoch"    : best_epoch,
                        "step"          : scheduler.last_epoch,
                        "model_info"    : model_info}
            
            # 保存当前最佳模型的权重
            best_model_path = f"{save_weights_path}/model_best_ep_{best_epoch}.pth"
            torch.save(save_file, best_model_path)
            print(f"✨Best model saved at epoch: {best_epoch} ✨with mean loss: {best_mean_loss}")
            
            # 删除之前保存的所有包含"model_best"的文件
            path_list = os.listdir(save_weights_path)
            for i in path_list:
                if "model_best" in i and i != f"model_best_ep_{best_epoch}.pth":
                    os.remove(os.path.join(save_weights_path, i))
                    print(f"✅remove last best weight:{i}")                         

            # 保存最后三个epoch权重
            if os.path.exists(f"{save_weights_path}/model_ep_{epoch-3}.pth"):
                os.remove(f"{save_weights_path}/model_ep_{epoch-3}.pth")
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)
            torch.save(save_file, f"{save_weights_path}/model_ep_{epoch}.pth") 
        
        # 记录验证loss是否出现上升       
        if val_mean_loss <= current_mean_loss:
            current_mean_loss = val_mean_loss 
            patience = 0   
        else:
            patience += 1 
    
        # 早停判断
        if patience >= 30:    
            print('恭喜你触发早停！！')
            break

    writer.close()
    total_time = time.time() - initial_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("✅training over. ⏳total time: {}".format(total_time_str))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train model on SEM stone dataset")
    
    # 保存路径
    parser.add_argument('--data_path',          type=str, 
                        default="/mnt/e/VScode/WS-Hub/Linux-md_seg/Medical_image_segmentation/medical_datasets/Synapse/CSV/synapse.csv", 
                        help="path to csv dataset")
    
    parser.add_argument('--data_root_path',  type=str,
                        default="/mnt/e/VScode/WS-Hub/Linux-md_seg/Medical_image_segmentation/medical_datasets/Synapse/CSV")
    
    # results
    parser.add_argument('--results_path',   type=str, 
                        default='/mnt/e/VScode/WS-Hub/Linux-md_seg/Medical_image_segmentation/results')
    
    parser.add_argument('--save_scores',   type=str, 
                        default='save_scores')
    
    parser.add_argument('--save_weight',   type=str,
                        default="save_weights")
    
    parser.add_argument('--log_name',  type=str,
                        default="logs")
    
    parser.add_argument('--modification', type=str,
                        default="modification_log")
    
    # 模型配置
    parser.add_argument('--model',              type=str, 
                        default="unet", 
                        help=" unet, ResD_unet, rdam_unet, a_unet, m_unet, aicunet\
                               Segnet, deeplabv3_resnet50, deeplabv3_mobilenetv3_large, pspnet, u2net_full, u2net_lite,")
    
    parser.add_argument('--loss_fn',            type=str, 
                        default='DiceLoss', 
                        help="'CrossEntropyLoss', 'FocalLoss', 'DiceLoss', 'WDiceLoss', 'DWDLoss', 'IoULoss', 'dice_hd'")
    
    parser.add_argument('--optimizer',          type=str, 
                        default='AdamW', 
                        help="'AdamW', 'SGD' or 'RMSprop'.")
    
    parser.add_argument('--scheduler',          type=str, 
                        default='CosineAnnealingLR', 
                        help="'CosineAnnealingLR', 'ReduceLROnPlateau'.")
    
    # 正则化
    parser.add_argument('--elnloss',        type=bool,  default=False)
    parser.add_argument('--l1_lambda',      type=float, default=0.001)
    parser.add_argument('--l2_lambda',      type=float, default=0.001)
    parser.add_argument('--dropout_p',      type=float, default=0.1  )
     
    parser.add_argument('--device',         type=str,   default='cuda:0')
    parser.add_argument('--resume',         type=str,   default=None,   help="the path of weight for resuming")
    parser.add_argument('--amp',            type=bool,  default=True,   help='use mixed precision training or not')
    
    # flag参数
    parser.add_argument('--tb',             type=bool,  default=True,   help='use tensorboard or not')   
    parser.add_argument('--save_flag',      type=bool,  default=True,   help='save weights or not')    
    parser.add_argument('--split_flag',     type=bool,  default=False,  help='split data or not')
    parser.add_argument('--change_params',  type=bool,  default=False,  help='change params or not')       
    
    # 训练参数
    parser.add_argument('--train_ratio',    type=float, default=0.7) 
    parser.add_argument('--val_ratio',      type=float, default=0.2)
    parser.add_argument('--batch_size',     type=int,   default=8  ) 
    parser.add_argument('--start_epoch',    type=int,   default=0,      help='start epoch')
    parser.add_argument('--end_epoch',      type=int,   default=200,    help='ending epoch')
    parser.add_argument('--warmup_epochs',  type=int,   default=10,      help='number of warmup epochs')


    parser.add_argument('--lr',             type=float, default=8e-4,   help='learning rate')
    parser.add_argument('--wd',             type=float, default=1e-4,   help='weight decay')
    
    parser.add_argument('--eval_interval',  type=int,   default=1,      help='interval for evaluation')
    parser.add_argument('--num_small_data', type=int,   default=None,   help='number of small data')
    parser.add_argument('--Tmax',           type=int,   default=60,     help='the numbers of half of T for CosineAnnealingLR')
    parser.add_argument('--eta_min',        type=float, default=1e-8,   help='minimum of lr for CosineAnnealingLR')

    main_args = parser.parse_args()

    aug_data_parser = argparse.ArgumentParser(description="data augmentation")
    aug_data_parser.add_argument("--root_path",    default="/mnt/e/VScode/WS-Hub/WS-UNet/UNet/datasets",            type=str,       help="root path")
    aug_data_parser.add_argument("--size",         default="256",                                                   type=str,       help="size of img and mask") 
    aug_data_parser.add_argument("--aug_times",    default=60,                                                      type=int,       help="augmentation times")
    aug_args = aug_data_parser.parse_args()

    main(main_args, aug_args)