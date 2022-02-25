import os
import sys
import gzip
import random
import platform
import warnings
import collections
from tqdm import tqdm, tqdm_notebook

import glob
import math
import cv2
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import transforms, utils
# from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import transforms


from model import Resnet18
from dataset import MyDataset




class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image = image)
            transformed_image = transformed['image']
        return transformed_image

    def __len__(self):
        return len(self.img_paths)


# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
def make_submission(model, loader, device, save_dir, name):
    model.eval()
    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    os.makedirs(save_dir, exist_ok=True)
    submission.to_csv(f'{save_dir}/{name}_submission.csv', index=False)
    print('test inference is done!')



# -- settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# -- load data
# 데이터셋 경로 지정
TEST_DATA_PATH = '/opt/ml/input/data/eval/info.csv'
TEST_IMG_PATH = '/opt/ml/input/data/eval/images'


# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(TEST_DATA_PATH)
image_paths = [os.path.join(TEST_IMG_PATH, img_id) for img_id in submission.ImageID]


# -- augmentation
transform = A.Compose([
    A.Resize(512, 384),
    A.Normalize(mean=0.5, std=0.2),
    transforms.ToTensorV2(transpose_mask=True)
    #A.CenterCrop(height=300, width=300, p=1)
])

# -- data_loader
# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
test_dataset = TestDataset(image_paths, transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# loss가 가장 낮았을때 모델 불러옴
resnet18_best_CEloss = torch.load(f"{model_saved}/loss_resnet18_CEloss_batchsize:{BATCH_SIZE}_lr:{LEARNING_RATE}.pt")
# acc가 가장 높았을때 모델 불러옴
resnet18_best_acc = torch.load(f"{model_saved}/accr_resnet18_CEloss_batchsize:{BATCH_SIZE}_lr:{LEARNING_RATE}.pt")

save_dir = './resnet18/'

make_submission(resnet18_best_CEloss, test_dataloader, device, save_dir, 'CEloss_base')
