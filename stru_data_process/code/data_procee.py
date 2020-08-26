# coidng: utf-8


"""
数据预处理部分，对于分割问题， 分别对数据进行增强操作

"""

import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
from utils import *

# Python图像处理库 - Albumentations，可用于深度学习中网络训练时的图片数据增强.
"""
Albumentations 图像数据增强库特点：

基于高度优化的 OpenCV 库实现图像快速数据增强.
针对不同图像任务，如分割，检测等，超级简单的 API 接口.
易于个性化定制.
易于添加到其它框架，比如 PyTorch.
"""

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)


# 读取图像信息
def read_data(img, lab):
    img_data = cv2.imread(img, 0)
    lab_data = cv2.imread(lab, 0)

    return img_data, lab_data


# 增强数据
def augment_data(images, masks, save_path, augment=True):
    """
    Performing data augmentation.
    """

    crop_size = (256, 256)
    size = (2018, 2006)
    # 将数据与标签组合
    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = image.split("/")[-1].split(".")[0]
        mask_name = mask.split("/")[-1].split(".")[0]

        x, y = read_data(image, mask)
        # try except 使用
        try:
            h, w, c = x.shape   # 获取图像的 w h z
        except Exception as e:
            image = image[:-1]
            x, y = read_data(image, mask)
            h, w, c = x.shape

        # 进行数据增强
        if augment == True:
            # Center Crop
            aug = CenterCrop(p=1, height=crop_size[1], width=crop_size[0])
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            # Crop
            x_min = 0
            y_min = 0
            x_max = x_min + crop_size[0]
            y_max = y_min + crop_size[1]

            aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            # Random Rotate 90 degree
            aug = RandomRotate90(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            # Transpose
            aug = Transpose(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            # ElasticTransform
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            # Grid Distortion
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            # Optical Distortion
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            # Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            # Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            # Grayscale
            x10 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y10 = y

            # Grayscale Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x11 = augmented['image']
            y11 = augmented['mask']

            # Grayscale Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x12 = augmented['image']
            y12 = augmented['mask']

            # Grayscale Center Crop
            aug = CenterCrop(p=1, height=crop_size[1], width=crop_size[0])
            augmented = aug(image=x10, mask=y10)
            x13 = augmented['image']
            y13 = augmented['mask']

            # Random Brightness Contrast
            aug = RandomBrightnessContrast(p=1)
            augmented = aug(image=x, mask=y)
            x14 = augmented['image']
            y14 = augmented['mask']

            # Random Gamma
            aug = RandomGamma(p=1)
            augmented = aug(image=x, mask=y)
            x15 = augmented['image']
            y15 = augmented['mask']

            aug = HueSaturationValue(p=1)
            augmented = aug(image=x, mask=y)
            x16 = augmented['image']
            y16 = augmented['mask']

            aug = RGBShift(p=1)
            augmented = aug(image=x, mask=y)
            x17 = augmented['image']
            y17 = augmented['mask']

            aug = RandomBrightness(p=1)
            augmented = aug(image=x, mask=y)
            x18 = augmented['image']
            y18 = augmented['mask']

            aug = RandomContrast(p=1)
            augmented = aug(image=x, mask=y)
            x19 = augmented['image']
            y19 = augmented['mask']

            aug = MotionBlur(p=1, blur_limit=7)
            augmented = aug(image=x, mask=y)
            x20 = augmented['image']
            y20 = augmented['mask']

            aug = MedianBlur(p=1, blur_limit=10)
            augmented = aug(image=x, mask=y)
            x21 = augmented['image']
            y21 = augmented['mask']

            aug = GaussianBlur(p=1, blur_limit=10)
            augmented = aug(image=x, mask=y)
            x22 = augmented['image']
            y22 = augmented['mask']

            aug = GaussNoise(p=1)
            augmented = aug(image=x, mask=y)
            x23 = augmented['image']
            y23 = augmented['mask']

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x24 = augmented['image']
            y24 = augmented['mask']

            aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x25 = augmented['image']
            y25 = augmented['mask']

            images = [
                x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                x21, x22, x23, x24, x25
            ]
            masks = [
                y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
                y21, y22, y23, y24, y25
            ]

        else:
            images = [x]
            masks = [y]

        idx = 0
        # 数据增强之后数据
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{image_name}_{idx}.jpg"
            tmp_mask_name = f"{mask_name}_{idx}.jpg"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path = os.path.join(save_path, "mask/", tmp_mask_name)

            # 保存数据
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


# 加载数据
def load_data(path, split=0.1):
    """ Load all the data and then split them into train and valid dataset. """
    img_path = glob(os.path.join(path, "images/*"))
    msk_path = glob(os.path.join(path, "masks/*"))
    print(len(img_path), len(msk_path))

    img_path.sort()
    msk_path.sort()

    total = len(img_path)
    # 训练集、验证集以及测试集划分
    train_size = int(0.8 * total)
    valid_size = int(0.1 * total)
    test_size = int(0.1 * total)

    """_train, X_test, y_train, y_test = train_test_split(
          ...     X, y, test_size=0.33, random_state=42)
    """
    train_x, test_x = train_test_split(img_path, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(msk_path, test_size=test_size, random_state=42)

    train_x, valid_x = train_test_split(train_x, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=valid_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def main():
    np.random.seed(42)
    # 数据路径
    path = "data/CVC-612/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path, split=0.1)

    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/valid/image/")
    create_dir("new_data/valid/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(valid_x, valid_y, "new_data/valid/", augment=False)
    augment_data(test_x, test_y, "new_data/test/", augment=False)


if __name__ == "__main__":
    main()

