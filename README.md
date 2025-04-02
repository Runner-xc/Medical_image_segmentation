# Unet系列
基于U-Net架构设计医学影像分割网络

## 项目文件结构
```
├── utils: 工具箱
├── SEM_Data.py:      自定义数据集读取相关代码
├── predict.py:       预测代码   
├── train.py:         GPU 或 CPU 训练代码
├── transforms.py:    数据预处理相关代码  
└── requirements.txt: 项目依赖
```
## 数据集结构
```
├── DATASETS
       ├── images: 该文件夹存放所有的图片
       ├── masks:  该文件夹存放所有的图片
       └── CSV:    该文件夹存放所有csv文件
            ├── DATASETS.csv: 总数据集的路径
            ├── TRAIN.csv:    训练数据集路径
            ├── VAL.csv:      验证数据集路径
            └── TEST.csv:     测试数据集路径
```
- 注意训练或者验证过程中，将`--data-path`指向`.csv`文件具体路径
