import torch
import datetime
from torch.utils.data import DataLoader
from utils.my_data import SEM_DATA
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
from model.rdam_unet import RDAM_UNet
from model.vm_unet import VMUNet
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

# ---------------------------- Constants and Presets ----------------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ConfigPreset:
    """数据预处理配置预设"""
    @staticmethod
    def train_preset(base_size, crop_size=256, hflip_prob=0.5):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @staticmethod
    def eval_preset(base_size):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

# ---------------------------- Core Components ----------------------------
class TrainingComponents:
    """训练组件"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
    def get_model(self):
        """模型"""
        model_map = {
            "u2net_full": u2net_full_config(),
            "u2net_lite": u2net_lite_config(),

            "unet": UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "ResD_unet": ResD_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "a_unet": A_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "m_unet": M_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "rdam_unet": RDAM_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),

            "Segnet": SegNet(n_classes=4, dropout_p=args.dropout_p),
            "pspnet": PSPNet(classes=4, dropout=args.dropout_p, pretrained=False),
            "deeplabv3_resnet50": deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=4),
            "deeplabv3_resnet101": deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=4),
            "deeplabv3_mobilenetv3_large": deeplabv3_mobilenetv3_large(aux=False, pretrain_backbone=False, num_classes=4)
        }
        model = model_map.get(self.args.model)
        if not model:
            raise ValueError(f"Invalid model name: {self.args.model}")
        kaiming_initial(model)
        return model.to(self.device)

    def get_optimizer(self, model):
        """优化器"""
        optim_map = {
            'AdamW': lambda: AdamW(model.parameters(), self.args.lr, weight_decay=self.args.wd),
            'SGD': lambda: SGD(model.parameters(), self.args.lr, momentum=0.9, weight_decay=self.args.wd),
            'RMSprop': lambda: RMSprop(model.parameters(), self.args.lr, alpha=0.9, eps=1e-8, weight_decay=self.args.wd)
        }
        return optim_map.get(self.args.optimizer, optim_map['AdamW'])()

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio   

    # 划分数据集
    if args.num_small_data is not None:
        train_datasets, val_datasets, test_datasets = data_split.small_data_split_to_train_val_test(args.data_path, 
                                                                                                    num_small_data=args.num_small_data, 
                                                                                                    # train_ratio=0.8, 
                                                                                                    # val_ratio=0.1, 
                            save_root_path=args.data_root_path,
                            flag=args.split_flag) 
    
    else:
        train_datasets, val_datasets, test_datasets = data_split.data_split_to_train_val_test(args.data_path, train_ratio=train_ratio, val_ratio=val_ratio,
                            save_root_path=args.data_root_path,   # 保存划分好的数据集路径
                            flag=args.split_flag)

    # 读取数据集
    train_datasets = SEM_DATA(train_datasets, 
                            transforms=SODPresetTrain((256, 256), crop_size=256))
    
    val_datasets = SEM_DATA(val_datasets, 
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
    if args.model =="u2net_full":
        model = u2net_full_config()
    elif args.model =="u2net_lite":
        model = u2net_lite_config()
    
    # unet系列
    elif args.model == "unet":   
        model = UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "ResD_unet":
        model = ResD_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "a_unet":
        model = A_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "m_unet":
        model = M_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)    
    elif args.model == "rdam_unet":
        model = RDAM_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "aicunet":
        model = AICUNet(in_channels=3, n_classes=4, base_channels=32, p=args.dropout_p)
    elif args.model == "vm_unet":
        model = VMUNet(input_channels=3, num_classes=4)
    
    # 其他模型        
    elif args.model == "Segnet":
        model = SegNet(n_classes=4, dropout_p=args.dropout_p)
    elif args.model == "pspnet":
        model = PSPNet(classes=4, dropout=args.dropout_p, pretrained=False)
    elif args.model == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=4)
    elif args.model == "deeplabv3_resnet101":
        model = deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=4)
    elif args.model == "deeplabv3_mobilenetv3_large":
        model = deeplabv3_mobilenetv3_large(aux=False, pretrain_backbone=False, num_classes=4)
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    
    # 初始化模型
    kaiming_initial(model)
    model.to(device)
    model_info = str(summary(model, (1, 3, 256, 256)))  
    
    """——————————————————————————————————————————————优化器 调度器——————————————————————————————————————————————"""
    # 优化器 
    assert args.optimizer in ['AdamW', 'SGD', 'RMSprop'], \
        f'optimizer must be AdamW, SGD, RMSprop but got {args.optimizer}'
        
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=args.wd
                          ) # 会出现梯度爆炸或消失

    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                        weight_decay=args.wd
                        )

    elif args.optimizer == 'RMSprop':

        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, 
                            weight_decay=args.wd
                            )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=args.wd
                          )
        
    # 调度器
    assert args.scheduler in ['CosineAnnealingLR', 'ReduceLROnPlateau'], \
            f'scheduler must be CosineAnnealingLR 、ReduceLROnPlateau, but got {args.scheduler}'
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
            last_epoch=-1  # 初始步数从0开始
        )
        
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
    assert args.loss_fn in ['CrossEntropyLoss', 'DiceLoss', 'FocalLoss', 'WDiceLoss', 'DWDLoss', 'IoULoss', 'dice_hd']
    if args.loss_fn == 'CrossEntropyLoss':
        loss_fn = CrossEntropyLoss()
    elif args.loss_fn == 'DiceLoss':
        loss_fn = diceloss()
    elif args.loss_fn == 'FocalLoss':
        loss_fn = Focal_Loss()
    elif args.loss_fn == 'WDiceLoss':
        loss_fn = WDiceLoss()
    elif args.loss_fn == 'DWDLoss':
        loss_fn = DWDLoss()
    elif args.loss_fn == 'IoULoss':
        loss_fn = IOULoss()
    elif args.loss_fn == 'dice_hd':
        loss_fn = AdaptiveSegLoss(num_classes=4)
    
    # 缩放器
    scaler = torch.amp.GradScaler() if args.amp else None
    Metrics = Evaluate_Metric()
    
    # 日志保存路径
    save_logs_path = f"{args.log_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}"
    
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
        excluded_params = ['data_path', 'data_root_path', 'save_scores_path', 
                         'save_weight_path', 'log_path', 'modification_path',
                         'device', 'resume', 'save_flag', 'split_flag', 'change_params']
        config = {k: v for k, v in config.items() if k not in excluded_params}
        
        # 初始化SwanLab（整个训练过程只初始化一次）
        swanlab.init(
            experiment_name=f"{args.model}-{args.loss_fn}",
            config=config,
            logdir=log_path  # 与TensorBoard共享日志目录
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
            
            return LambdaLR(optimizer, lambda step: (
                (step / warmup_steps) if step < warmup_steps
                else (self.args.eta_min + (lr_initial - self.args.eta_min) * 
                    (1 + math.cos(math.pi * (step - warmup_steps) / Tmax_steps)) / 2) / lr_initial
            ), last_epoch=-1)
            
        elif self.args.scheduler == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")

    def get_loss_fn(self):
        """损失函数"""
        loss_map = {
            'CrossEntropyLoss': CrossEntropyLoss(),
            'DiceLoss': diceloss(),
            'FocalLoss': Focal_Loss(),
            'WDiceLoss': WDiceLoss(),
            'DWDLoss': DWDLoss(),
            'IoULoss': IOULoss(),
            'dice_hd': AdaptiveSegLoss(4)
        }
        return loss_map.get(self.args.loss_fn)

    def get_writer(self, detailed_time_str):
        # 日志保存路径
        save_logs_path = f"{self.args.log_path}/{self.args.model}/L_{self.args.loss_fn}--S_{self.args.scheduler}"
        
        if not os.path.exists(save_logs_path):
            os.makedirs(save_logs_path)
        
        if self.args.elnloss:
            log_path = f'{save_logs_path}/optim_{self.args.optimizer}-lr_{self.args.lr}-l1_{self.args.l1_lambda}-l2_{self.args.l2_lambda}/{detailed_time_str}'
            return SummaryWriter(log_path), log_path
        else:
            log_path = f'{save_logs_path}/optim_{self.args.optimizer}-lr_{self.args.lr}-wd_{self.args.wd}/{detailed_time_str}'
            return SummaryWriter(log_path), log_path
        

# ---------------------------- Data Management ----------------------------
class DataManager:
    """数据管理"""
    def __init__(self, args):
        self.args = args
        self.base_transform = ConfigPreset()

    def load_datasets(self):
        """加载数据集"""
        if self.args.num_small_data:
            train, val, test = data_split.small_data_split_to_train_val_test(
                self.args.data_path, self.args.num_small_data, self.args.split_flag, self.args.train_ratio, self.args.val_ratio,self.args.data_root_path)
        else:
            train, val, test = data_split.data_split_to_train_val_test(
                self.args.data_path, self.args.split_flag, self.args.train_ratio, self.args.val_ratio, 
                self.args.data_root_path)

        train_set = SEM_DATA(train, self.base_transform.train_preset(256))
        val_set = SEM_DATA(val, self.base_transform.eval_preset(256))
        return train_set, val_set

    def get_dataloaders(self, train_set, val_set):
        """获取数据加载器"""
        num_workers = min(os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8)
        train_loader = DataLoader(train_set, self.args.batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, 8, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

# ---------------------------- Training Logic ----------------------------
class TrainingManager:
    """训练流程管理"""
    def __init__(self, args, model, optimizer, scheduler, loss_fn, device):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = Evaluate_Metric()
        self.scaler = torch.amp.GradScaler() if args.amp else None
     
    def _train_epoch(self, epoch, dataloader):
        """
        训练单个epoch
        """
        start_time = time.time()
        components_dict = {"model"      :self.model, 
                           "optimizer"  :self.optimizer, 
                           "epoch"      :epoch, 
                           "dataloader" :dataloader, 
                           "device"     :self.device, 
                           "loss_fn"    :self.loss_fn, 
                           "scaler"     :self.scaler,
                           "metrics"    :self.metrics,
                           "scheduler"  :self.scheduler,
                           "elnloss"    :self.args.elnloss, 
                           "l1_lambda"  :self.args.l1_lambda,
                           "l2_lambda"  :self.args.l2_lambda}
        # results = train_loss, T_OM_loss, T_OP_loss, T_IOP_loss, T_Metric_list
        results = train_one_epochv2(components_dict)
        loss = [x / len(dataloader) for x in results[:-1]]
        self.train_mean_loss = loss[3]
        results_dict = {"Loss"      : loss, 
                        "Recall"    :results[-1][0], 
                        "Precision" :results[-1][1], 
                        "Dice"      :results[-1][2], 
                        "F1_scores" :results[-1][3], 
                        "mIoU"      :results[-1][4], 
                        "Accuracy"  :results[-1][5]}
        names = ["Loss", "Recall", "Precision", "Dice", "F1_scores", "mIoU", "Accuracy"]
        train_metrics = {f"{name}" : results_dict[name] for name in names}

        end_time = time.time()
        train_cost_time = end_time - start_time
        print(
              f"train_OM_loss: {loss[0]:.3f}\n"
              f"train_OP_loss: {loss[1]:.3f}\n"
              f"train_IOP_loss: {loss[2]:.3f}\n"
              f"train_mean_loss: {loss[3]:.3f}\n"
              f"train_cost_time: {train_cost_time:.2f}s\n")
        return train_metrics

    def _validate_epoch(self,val_dataloader):
        """
        验证单个epoch
        """
        start_time = time.time()
        # results = OM_loss,OP_loss,IOP_loss, mean_loss, Metric_list
        results = evaluate(self.model, self.device, val_dataloader, self.loss_fn, self.metrics)
        loss = [x / len(val_dataloader) for x in results[:-1]]
        self.val_mean_loss = loss[3]
        results_dict = {"Loss"      : loss, 
                        "Recall"    : results[-1][0], 
                        "Precision" : results[-1][1], 
                        "Dice"      : results[-1][2], 
                        "F1_scores" : results[-1][3], 
                        "mIoU"      : results[-1][4], 
                        "Accuracy"  : results[-1][5]}
        names = ["Loss", "Recall", "Precision", "Dice", "F1_scores", "mIoU", "Accuracy"]
        self.val_metrics = {f"{name}" : results_dict[name] for name in names}
        # 获取当前学习率
        current_lr = self.scheduler.get_last_lr()[0]
        # 结束时间
        end_time = time.time()
        self.val_cost_time = end_time - start_time
        print(
            f"val_OM_loss: {loss[0]:.3f}\n"
            f"val_OP_loss: {loss[1]:.3f}\n"
            f"val_IOP_loss: {loss[2]:.3f}\n"
            f"val_mean_loss: {loss[3]:.3f}\n"
            f"val_cost_time: {self.val_cost_time:.2f}s\n")
        print(f"Current learning rate: {current_lr}\n")
        return self.val_metrics, loss[3]
    
    def run_logging(self,writer, train_metrics, val_metrics, epoch):
        if self.args.tb:
            writing_logs(writer, train_metrics, val_metrics, epoch)
    
    def save_metrics(self, args, epoch, end_epoch, best_epoch):
        metrics_table_header    = ['Metrics_Name', 'Mean', 'OM', 'OP', 'IOP']
        metrics_table_left      = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']
        epoch_s                 = f"✈✈✈✈✈ epoch : {epoch + 1} / {end_epoch} ✈✈✈✈✈✈\n"
        model_s                 = f"model : {args.model} \n"
        lr_s                    = f"lr : {args.lr} \n"
        wd_s                    = f"wd : {args.wd} \n"  #####
        dropout_s               = f"dropout : {args.dropout_p} \n"
        l1_lambda               = f"l1_lambda : {args.l1_lambda} \n"
        l2_lambda               = f"l2_lambda : {args.l2_lambda} \n"
        scheduler_s             = f"scheduler : {args.scheduler} \n"
        loss_fn_s               = f"loss_fn : {args.loss_fn} \n"
        best_epoch_s            = f"best_epoch : {best_epoch} \n"
        time_s                  = f"time : {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} \n"
        cost_s                  = f"cost_time :{self.val_cost_time / 60:.2f}mins \n"
        
        metrics_dict    = {scores : self.val_metrics[scores] for scores in metrics_table_left}
        metrics_table   = [[metric_name,
                            metrics_dict[metric_name][-1],
                            metrics_dict[metric_name][0],
                            metrics_dict[metric_name][1],
                            metrics_dict[metric_name][2]
                        ]
                            for metric_name in metrics_table_left
                        ]
        table_s         = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
        train_loss_s    = f"train_loss : {self.train_mean_loss:.3f}  🍎🍎🍎\n"
        loss_s          = f"val_loss : {self.val_mean_loss:.3f}   🍎🍎🍎\n"

        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        write_info      = epoch_s + model_s + lr_s + wd_s + dropout_s + l1_lambda + l2_lambda + loss_fn_s + scheduler_s + train_loss_s + loss_s + table_s + '\n' + best_epoch_s + cost_s + time_s
        print(write_info)

        # 保存结果
        save_scores_path = f'{args.save_scores_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}'
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
    
    def save_weights(self, args, epoch, best_mean_loss, best_epoch, model_info):
        if args.save_flag:
            # 保存best模型
            if args.elnloss:
                save_weights_path = f"{args.save_weight_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}/optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}"  # 保存权重路径
            else:
                save_weights_path = f"{args.save_weight_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}/optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}"
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)

            save_file = {"model"        : self.model.state_dict(),
                        "optimizer"     : self.optimizer.state_dict(),
                        "Metrics"       : self.metrics.state_dict(),
                        "scheduler"     : self.scheduler.state_dict(),
                        "best_mean_loss": best_mean_loss,
                        "best_epoch"    : best_epoch,
                        "step"          : self.scheduler.last_epoch,
                        "model_info"    : model_info}
            
            # 保存当前最佳模型的权重
            best_model_path = f"{save_weights_path}/model_best_ep_{best_epoch}.pth"
            torch.save(save_file, best_model_path)
            print(f"Best model saved at epoch {best_epoch} with mean loss {best_mean_loss}")
            # 删除之前保存的所有包含"model_best"的文件
            path_list = os.listdir(save_weights_path)
            for i in path_list:
                if "model_best" in i and i != f"model_best_ep_{best_epoch}.pth":
                    os.remove(os.path.join(save_weights_path, i))
                    print(f"remove last best weight:{i}")
                    
            # only save latest 3 epoch weights
            if os.path.exists(f"{save_weights_path}/model_ep_{epoch-3}.pth"):
                os.remove(f"{save_weights_path}/model_ep_{epoch-3}.pth")
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)
            torch.save(save_file, f"{save_weights_path}/model_ep_{epoch}.pth") 

# ---------------------------- Main Function ----------------------------
def main(args, detailed_time_str):
    """主训练流程"""
    # 初始化组件
    components = TrainingComponents(args)
    data_mgr = DataManager(args)
    
    # 加载数据
    train_set, val_set = data_mgr.load_datasets()
    train_loader, val_loader = data_mgr.get_dataloaders(train_set, val_set)
    
    # 初始化模型和训练组件
    model = components.get_model()
    model_info = str(summary(model, (1, 3, 256, 256)))

    optimizer = components.get_optimizer(model)
    scheduler = components.get_scheduler(optimizer, train_loader)
    loss_fn = components.get_loss_fn()
    if args.save_flag:
        writer, log_path = components.get_writer(detailed_time_str)
    
    # 初始化训练管理器
    trainer = TrainingManager(args, model, optimizer, scheduler, loss_fn, components.device)

    best_mean_loss, current_miou = float('inf'), 0.0
    best_epoch = 0 
    patience = 0 
    current_mean_loss = float('inf')
    start_epoch = args.start_epoch
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
        print(f"Resume from epoch: {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, args.end_epoch):
        print(f"\n✈✈✈ Epoch {epoch+1}/{args.end_epoch} ✈✈✈")
        
        # 训练阶段
        train_metrics = trainer._train_epoch(epoch, train_loader)
        
        # 验证阶段
        if epoch % args.eval_interval == 0 or epoch == args.end_epoch - 1:
            val_metrics, val_mean_loss = trainer._validate_epoch(val_loader)
            
            # 记录日志
            trainer.run_logging(writer, train_metrics, val_metrics, epoch)
            if epoch == 5:
                    run_tensorboard(log_path)

            if best_mean_loss >= val_mean_loss:
                best_mean_loss = val_mean_loss
                best_epoch = epoch + 1
                
            # 保存指标
            trainer.save_metrics(args, epoch, args.end_epoch, best_epoch)
            # 保存权重
            trainer.save_weights(args, epoch, best_mean_loss, best_epoch, model_info)

        # 记录验证loss是否出现上升       
        if val_mean_loss <= current_mean_loss:
            current_mean_loss = val_mean_loss 
            patience = 0   
        else:
            patience += 1
        # 早停检查
        if patience >= 50:
            print("Early stopping triggered!")
            break

    # 清理资源
    writer.close()
    print(f"Training completed in {time.time()-detailed_time_str:.2f} seconds")

def parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser(description="SEM图像分割训练脚本")
    # 保存路径
    parser.add_argument('--data_path',          type=str, 
                        default="/root/projects/WS-U2net/U-2-Net/datasets/CSV/rock_sem_chged_256_a50_c80.csv", 
                        help="path to csv dataset")
    
    parser.add_argument('--data_root_path',  type=str,
                        default="/root/projects/WS-U2net/U-2-Net/datasets/CSV")
    
    # results
    parser.add_argument('--save_scores_path',   type=str, 
                        default='/root/projects/WS-U2net/U-2-Net/results/save_scores')
    
    parser.add_argument('--save_weight_path',   type=str,
                        default="/root/projects/WS-U2net/U-2-Net/results/save_weights")
    
    parser.add_argument('--log_path',  type=str,
                        default="/root/projects/WS-U2net/U-2-Net/results/logs")
    
    parser.add_argument('--modification_path', type=str,
                        default="/root/projects/WS-U2net/U-2-Net/results/modification_log")
    
    # 模型配置
    parser.add_argument('--model',              type=str, 
                        default="rdam_unet", 
                        help=" unet, ResD_unet, rdam_unet, a_unet, m_unet, aicunet, vm_unet\
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
    parser.add_argument('--dropout_p',      type=float, default=0.4  )
     
    parser.add_argument('--device',         type=str,   default='cuda:0')
    parser.add_argument('--resume',         type=str,   default=None,   help="the path of weight for resuming")
    parser.add_argument('--amp',            type=bool,  default=True,   help='use mixed precision training or not')
    
    # flag参数
    parser.add_argument('--tb',             type=bool,  default=False,   help='use tensorboard or not')   
    parser.add_argument('--save_flag',      type=bool,  default=False,   help='save weights or not')    
    parser.add_argument('--split_flag',     type=bool,  default=False,  help='split data or not')
    parser.add_argument('--change_params',  type=bool,  default=False,  help='change params or not')       
    
    # 训练参数
    parser.add_argument('--train_ratio',    type=float, default=0.7) 
    parser.add_argument('--val_ratio',      type=float, default=0.1)
    parser.add_argument('--batch_size',     type=int,   default=8  ) 
    parser.add_argument('--start_epoch',    type=int,   default=0,      help='start epoch')
    parser.add_argument('--end_epoch',      type=int,   default=200,    help='ending epoch')
    parser.add_argument('--warmup_epochs',  type=int,   default=10,      help='number of warmup epochs')


    parser.add_argument('--lr',             type=float, default=8e-4,   help='learning rate')
    parser.add_argument('--wd',             type=float, default=1e-6,   help='weight decay')
    
    parser.add_argument('--eval_interval',  type=int,   default=1,      help='interval for evaluation')
    parser.add_argument('--num_small_data', type=int,   default=None,   help='number of small data')
    parser.add_argument('--Tmax',           type=int,   default=45,     help='the numbers of half of T for CosineAnnealingLR')
    parser.add_argument('--eta_min',        type=float, default=1e-8,   help='minimum of lr for CosineAnnealingLR')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.change_params:
        args = param_modification.param_modification(args)

    detailed_time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
    main(args, detailed_time_str)
