import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse

class ShaleAugmentor:
    def __init__(self, config=None):
        # 默认配置
        default_config = {
            "unsharp_mask": {"p": 0.2, "sigma": 1.0, "strength": 0.5},
            "morphology": {"kernel_size": 3, "p": 0.3},
            "histogram_equalization": {"p": 0.05},                              # 降低直方图均衡化的概率
            "noise": {"gaussian_p": 0.3, "sp_p": 0.05, "gaussian_sigma": 40},   # 降低噪声添加的概率和强度
            "random_erase": {"p": 0.2, "erase_area_ratio": 0.01},               # 降低擦除的概率和区域大小
            "sobel": {"threshhold" : 120, "p": 0.2},
            "random_rotate": {"p": 0.3},    
        }
        # 深度合并配置
        if config:
            self.config = self._deep_update(default_config, config)
        else:
            self.config = default_config
        
        self.aug_methods = [
            # self.sobel,
            self.random_rotate,
            self.unsharp_mask,
            self.morphology_aug,
            # self.histogram_equalization,
            self.add_noise,      
            # self.random_erase,
        ]

    def __call__(self, image, mask):
        """
        执行数据增强流水线
        """
        # 转换为numpy数组以加速处理
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        
        # 随机执行增强方法
        for method in self.aug_methods:
            if np.random.rand() < self._get_prob(method.__name__):
                image, mask = method(image, mask)

        # 确保图像数据类型为 uint8
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        return Image.fromarray(image), Image.fromarray(mask)

    def _deep_update(self, target, src):
        """深度合并字典"""
        for k, v in src.items():
            if isinstance(v, dict):
                node = target.setdefault(k, {})
                self._deep_update(node, v)
            else:
                target[k] = v
        return target
    
    def _get_prob(self, method_name):
        """获取各方法的默认概率"""
        prob_map = {
            "random_rotate": self.config["random_rotate"]["p"],
            "unsharp_mask": self.config["unsharp_mask"]["p"],
            "sobel": self.config["sobel"]["p"],
            "morphology_aug": self.config["morphology"]["p"],
            "histogram_equalization": self.config["histogram_equalization"]["p"],
            "add_noise": self.config["noise"]["gaussian_p"],
            "random_erase": self.config["random_erase"]["p"]
        }
        return prob_map.get(method_name, 0.5)
    
    def sobel(self, image, mask):
        """Sobel边缘检测"""
        if np.random.randn() < 0.5:
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradmag = cv2.magnitude(sobelx, sobely)
            gradmag_normalized = (gradmag / gradmag.max() * 255).astype(np.uint8)
            image = cv2.addWeighted(image, 0.8, gradmag_normalized, 0.2, 0)  # 叠加边缘增强  
        return image, mask

    def morphology_aug(self, image, mask):
        """形态学增强"""
        if np.random.rand() < 0.5:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config["morphology"]["kernel_size"], self.config["morphology"]["kernel_size"]))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config["morphology"]["kernel_size"], self.config["morphology"]["kernel_size"]))
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image, mask

    def histogram_equalization(self, image, mask):
        """
        灰度图像直方图均衡化
        使用 OpenCV 的 cv2.equalizeHist 方法增强图像对比度。
        """
        if len(image.shape) != 2:
            raise ValueError("输入图像应为灰度图，但图像通道数为：{}".format(len(image.shape)))

        # 应用直方图均衡化
        image = cv2.equalizeHist(image)

        return image, mask

    def add_noise(self, image, mask):
        """添加噪声"""
        # 高斯噪声
        if np.random.rand() < self.config["noise"]["gaussian_p"]:
            noise = np.random.normal(0, self.config["noise"]["gaussian_sigma"], image.shape)  # 减小噪声强度
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # 椒盐噪声
        if np.random.rand() < self.config["noise"]["sp_p"]:
            # 椒盐噪声概率
            sp_prob = self.config["noise"]["sp_p"] / 2  # 将总概率均分给盐和椒
            if len(image.shape) == 3:
                salt_pepper = np.random.choice(
                    [0, 255],
                    size=image.shape[:2],
                    p=[sp_prob, 1 - sp_prob]
                )
                salt_pepper = salt_pepper[..., None]  # HxWx1
                salt_pepper = np.repeat(salt_pepper, 3, axis=2)  # HxWx3
            else:
                salt_pepper = np.random.choice(
                    [0, 255],
                    size=image.shape,
                    p=[sp_prob, 1 - sp_prob]
                )
            image = np.where(salt_pepper == 0, salt_pepper, image)
        image = image.astype(np.uint8)
        return image, mask
    
    def random_rotate(self, image, mask):
        """随机旋转"""
        angle = np.random.choice([0, 90, 180, 270]) if np.random.rand() < 0.3 \
               else np.random.uniform(-30, 30)
        
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return image, mask

    def random_erase(self, image, mask):
        """随机擦除（黑色区域）"""
        h, w = image.shape[:2]
        erase_area = self.config["random_erase"]["erase_area_ratio"] * h * w
        aspect_ratio = np.random.uniform(0.5, 2)
        
        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(erase_area / erase_h)
        
        if erase_h < h and erase_w < w:
            x = np.random.randint(0, w - erase_w)
            y = np.random.randint(0, h - erase_h)
            
            # 修改此处：填充黑色（0）
            if len(image.shape) == 3:
                image[y:y+erase_h, x:x+erase_w] = 0  # RGB 黑色 (0,0,0)
            else:
                image[y:y+erase_h, x:x+erase_w] = 0  # 灰度图黑色
            
            mask[y:y+erase_h, x:x+erase_w] = 0
        return image, mask

    def unsharp_mask(self, image, mask, sigma=1.5, strength=0.8):
        """
        Apply unsharp masking to an image.
        :param image: Input image (PIL Image or numpy array)
        :param sigma: Standard deviation for Gaussian blur
        :param strength: Strength of sharpening
        """
        sigma = self.config["unsharp_mask"]["sigma"]       # 需要修复此处
        strength = self.config["unsharp_mask"]["strength"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        arr = np.array(image).astype(np.float32)
        blurred = gaussian_filter(arr, sigma=sigma)
        masked = arr - blurred
        
        # Apply the mask to the original image
        sharpened_image = np.clip(arr + strength * masked, 0, 255).astype(np.uint8)
        
        return sharpened_image, mask
    
    @staticmethod
    def visualize(orig, aug):
        """可视化对比"""
        fig, axes = plt.subplots(2, 2, figsize=(12,12))
        
        axes[0,0].imshow(orig[0], cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[0,1].imshow(orig[1], cmap='jet', vmin=0, vmax=3)
        axes[0,1].set_title('Original Mask')
        
        axes[1,0].imshow(aug[0], cmap='gray')
        axes[1,0].set_title('Augmented Image')
        axes[1,1].imshow(aug[1], cmap='jet', vmin=0, vmax=3)
        axes[1,1].set_title('Augmented Mask')
        
        plt.tight_layout()
        plt.show()

def aug_data_processing(root_path, aug_times=60, data_csv=None, datasets_name=None, csv_name=None):
    path_dict = {'img': [], 'mask': []}
    # 配置参数
    config = {
        "output": {
            "images": os.path.join(root_path, "aug_results/images"),
            "masks": os.path.join(root_path, "aug_results/masks")
        },
        "aug_times": aug_times,
        }
    
    train_aug_config = {
                    "sobel": {"threshhold" : 110, "p": 0.2},
                    "unsharp_mask": {"p": 0.3, "sigma": 1.0, "strength": 0.5},  
                    "morphology": {"kernel_size": 5, "p": 0.3},
                    "histogram_equalization": {"p": 0.05},
                    "noise": {"gaussian_p": 0.5, "sp_p": 0.05, "gaussian_sigma": 15},
                    "random_erase": {"p": 0.4, "erase_area_ratio": 0.05},
                    "random_rotate": {"p": 0.5},
                }
    
    val_aug_config = {
                    "sobel": {"p": 0},
                    "unsharp_mask": {"p": 0},  
                    "morphology": {"kernel_size": 3, "p": 0.3},
                    "histogram_equalization": {"p": 0.02},
                    "noise": {"gaussian_p": 0.2, "sp_p": 0.1, "gaussian_sigma": 15},
                    "random_erase": {"p": 0.2, "erase_area_ratio": 0.01},
                    "random_rotate": {"p": 0.5},
                }
    
    test_aug_config = {
                    "sobel": {"p": 0},
                    "random_rotate": {"p": 0.5},
                    "unsharp_mask": {"p": 0},
                    "morphology": {"p": 0},
                    "histogram_equalization": {"p": 0},
                    "noise": {"gaussian_p": 0},
                    "random_erase": {"p": 0}
                }
    

    # 初始化数据集增强器
    train_augmentor = ShaleAugmentor(train_aug_config)
    val_augmentor = ShaleAugmentor(val_aug_config)
    test_augmentor = ShaleAugmentor(test_aug_config)
    if datasets_name == "train":
        augmentor = train_augmentor
    elif datasets_name == "val":
        augmentor = val_augmentor
    elif datasets_name == "test":
        augmentor = test_augmentor
    else:
        raise ValueError("Invalid datasets_name. Choose from 'train', 'val', or 'test'.")

    # 输出路径
    output_images_path = os.path.join(config["output"]["images"], "time_" + str(config['aug_times']))
    output_masks_path = os.path.join(config["output"]["masks"], "time_" + str(config['aug_times']))
    if datasets_name:
        output_images_path = os.path.join(output_images_path, datasets_name)
        output_masks_path = os.path.join(output_masks_path, datasets_name)
    print("Output images path:", output_images_path)
    print("Output masks path:", output_masks_path)
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)

    if not data_csv.empty:
        data_list = data_csv.tolist()

    else:    
        print("请上传数据集路径")

    # 处理每个样本
    for data in tqdm(data_list):
        if not isinstance(data, str):
            continue
        # 读取数据
        base_name = os.path.splitext(os.path.basename(data))[0]
        # try:
        image = Image.fromarray(np.load(data)["image"]*255).convert("L")
        mask = Image.fromarray(np.load(data)["label"])
        
        # 数据增强
        for i in range(config["aug_times"]):
            aug_img, aug_mask = augmentor(image, mask)
            
            # 保存结果
            img_path = os.path.join(output_images_path, f"{base_name}_aug{i}.jpg")
            mask_path = os.path.join(output_masks_path, f"{base_name}_aug{i}.png")
            aug_img.save(img_path)
            aug_mask.save(mask_path)
            path_dict["img"].append(img_path)
            path_dict["mask"].append(mask_path)
    
    return path_dict


def main(args):
    aug_data_processing(args.root_path, 
                        args.size, 
                        args.aug_times
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data augmentation")
    parser.add_argument("--root_path",    default="/mnt/e/VScode/WS-Hub/WS-UNet/UNet/datasets",      type=str,       help="root path")
    parser.add_argument("--size",         default="256",                                             type=str,       help="size of img and mask") 
    parser.add_argument("--aug_times",    default=60,                                                type=int,       help="augmentation times")
    args = parser.parse_args()
    main(args)
