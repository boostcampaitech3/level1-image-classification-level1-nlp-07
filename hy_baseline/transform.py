import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import transforms
import cv2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

def My_transform(mode='train'):
    
    transform_train_strong = Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=9, p=0.2),
            Blur(blur_limit=10, p=0.2),
        ], p=0.3),
        A.Cutout(num_holes=20, max_h_size=10, max_w_size=10, p=0.3),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=10, p=0.3),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.2),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        OneOf([
            CLAHE(clip_limit=5),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ])


    transform_train_origin = A.Compose([
                A.CenterCrop(height=150, width=150, p=0.2),
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=5,p=0.1),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(p=1.0), 
                    A.MedianBlur(blur_limit=5, p=1),  
                    A.Blur(blur_limit=3, p=1), 
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(p=1),
                    A.IAAEmboss(p=1),
                    A.RandomBrightnessContrast(p=1),            
                ], p=0.3),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),         
            ])

    transform_train = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
                transform_train_strong,
                transform_train_origin,
                ], p=0.6),
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
        transforms.ToTensorV2(transpose_mask=True)])


    transform_test = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
        transforms.ToTensorV2(transpose_mask=True),
    ])
    
    if mode=='train':
        return transform_train
    else:
        return transform_test
