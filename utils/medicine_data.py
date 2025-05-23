import os
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.data_augment import aug_data_processing
"""
1. save_npz_paths_to_csv
2. 读取数据集，返回张量
"""

def save_npz_paths_to_csv(root_path, csv_name) -> pd.DataFrame:
    """
    root_path : 数据集根目录(train、test)
    csv_name  : 保存的csv路径文件名称
    """
    path_dict = {'train': [], 'test': []}

    # 遍历root_path下的所有目录和文件
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name.startswith('train_npz'):
                dir_path = os.path.join(root, dir_name)
                print(f"当前目录名称：{dir_path}")

                # 准备读取数据，检查是否存在空mask
                data_path = os.listdir(dir_path)
                for i in data_path:
                    data = np.load(os.path.join(dir_path, i))
                    if data['label'].any() != 0:
                        path_dict['train'].append(os.path.join(dir_path, i))

            elif dir_name.startswith('test_npz'):
                dir_path = os.path.join(root, dir_name)
                print(f"当前目录名称：{dir_path}")

                data_path = os.listdir(dir_path)
                for i in data_path:
                    data = np.load(os.path.join(dir_path, i))
                    if data['label'].any() != 0:
                        path_dict['test'].append(os.path.join(dir_path, i))
    
    # 找到最长的数组长度
    max_length = max(len(value) for value in path_dict.values())

    # 填充较短的数组
    for key, value in path_dict.items():
        if len(value) < max_length:
            path_dict[key] = value + [None] * (max_length - len(value))

    # 保存DataFrame到CSV文件
    df = pd.DataFrame(path_dict)
    if not os.path.isdir(root_path):
        os.makedirs(root_path)
    csv_file_path = f"{root_path}/{csv_name}"
    df.to_csv(csv_file_path, index=False) 
    return df

def get_data_augmentation(df_path, root_path, aug_times=6):
    """
    root_path : 数据集根目录(train、test)
    df : 数据集 -> pd.DataFrame
    """
    # 读取数据集
    data_df = pd.read_csv(df_path)
    df_name = os.path.basename(df_path).split('.')[0]
    l = len(data_df["train"])
    train_df =  data_df["train"].iloc[:int(l * 0.7)]
    val_df = data_df["train"].iloc[int(l * 0.7):]
    test_df = data_df["test"]

    # 进行数据增强
    train_data_dict = aug_data_processing(root_path=root_path, aug_times=aug_times, data_csv=train_df, datasets_name="train", csv_name=df_name)
    val_data_dict = aug_data_processing(root_path=root_path, aug_times=aug_times, data_csv=val_df, datasets_name="val", csv_name=df_name)
    test_data_dict = aug_data_processing(root_path=root_path, aug_times=aug_times, data_csv=test_df, datasets_name="test", csv_name=df_name)
    
    # 保存数据集
    train_data = pd.DataFrame(train_data_dict)
    val_data = pd.DataFrame(val_data_dict)
    test_data = pd.DataFrame(test_data_dict)
    train_data.to_csv(f"{root_path}/CSV/train_{os.path.basename(df_path).split('.')[0]}.csv", index=False)
    val_data.to_csv(f"{root_path}/CSV/val_{os.path.basename(df_path).split('.')[0]}.csv", index=False)
    test_data.to_csv(f"{root_path}/CSV/test_{os.path.basename(df_path).split('.')[0]}.csv", index=False)
    print("数据集划分完成！")
    return train_data_dict, val_data_dict, test_data_dict

class Synapse_data(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self):
        df = pd.read_csv(self.data_path)
        return len(df)
    
    def __getitem__(self, index):
        data = self.load_data(index=index)
        return data
    
    # 加载数据集
    def load_data(self, index):
        
        df = pd.read_csv(self.data_path)
        img_list = df['img'].tolist()
        mask_list = df['mask'].tolist()
        img_path = img_list[index]
        mask_path = mask_list[index]
        
        # 确保路径存在
        assert os.path.exists(img_path), f"The image path '{img_path}' does not exist."
        assert os.path.exists(mask_path), f"The mask path '{mask_path}' does not exist."
        
        # 读取数据 
        img = Image.open(img_path).convert('RGB')     # 将图片转换为RGB格式
        mask = Image.open(mask_path).convert('L')     # 将mask转换为灰度图像, 不然会多一个维度
        img_array = np.array(img) 
        img_name = os.path.basename(img_path)
        # # 归一化图片
        img_array = img_array / 255.0 
        img_array = img_array.astype(np.float32)
        mask_array = np.array(mask)

        if self.transforms is not None:
            data = self.transforms(img_array, mask_array) 
            
        return data, img_name

if __name__ == '__main__':
    root_path = "/mnt/e/VScode/WS-Hub/Linux-md_seg/Medical_image_segmentation/medical_datasets/Synapse"
    csv_name = "/mnt/e/VScode/WS-Hub/Linux-md_seg/Medical_image_segmentation/medical_datasets/Synapse/CSV/synapse.csv"
    # save_npz_paths_to_csv(root_path, csv_name)
    get_data_augmentation(df_path=csv_name, root_path=root_path)