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


def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
set_seed(seed)


# 디렉토리 이름만 가지고 라벨링, 마스크여부랑 상관없이 gender, age 만 사용해 라벨링
def dir_labeling(name):
    label = 0
    info = name.split('_')
    gender, age = info[1], int(info[3])
    if gender == 'female':
        label += 3
    
    if 30 <= age < 60:
        label += 1
    elif age >= 60:
        label += 2

    return label


# dataframe path의 모든 하위 파일 찾기
def make_each_path(folder_path, dirs):
    result = []
    for image_path in list(dirs['dir_path']):
        d_name = os.path.join(folder_path, image_path)
        filenames = os.listdir(d_name)
        for filename in filenames:
            if filename[0] == '.':
                continue
            full_file_name = os.path.join(d_name, filename)
            result.append(full_file_name)
    return result


# 18개 클래스에 대해 label 생성    
def labeling(name):
    label = 0
    info, mask_type = name.split('/')[-2:]
    info = info.split('_')
    gender, age = info[1], int(info[3])
    if 'incorrect' in mask_type:
        label += 6
    elif 'normal' in mask_type:
        label += 12
    
    if gender == 'female':
        label += 3
        
    if 30 <= age < 60:
        label += 1
    elif age >= 60:
        label += 2
    
    return label


# 하위 디렉토리의 모든 파일 찾기(사용X)
def search(dirname, result):
    try:
        files = os.listdir(dirname)
        for file in files:
            if file[0] == '.':  # .으로 시작하는 파일은 버리기
                continue
            full_name = os.path.join(dirname, file)
            if os.path.isdir(full_name):
                search(full_name, result)
            else:
                ext = os.path.splitext(full_name)[-1]  # 확장자 확인
                if ext:
                    result.append(full_name)
    except PermissionError:
        pass



### 데이터 전처리 함수
def data_processing(TRAIN_DATA_PATH, TRAIN_IMG_PATH):
    ### 1. 데이터 불러와서 사람별로 split 
    # TRAIN_DATA_PATH = 'opt/ml/input/data/train/train.csv'
    # TRAIN_IMG_PATH = 'opt/ml/input/data/train/images'

    # 원본 csv를 dataframe으로 만들기
    df = pd.read_csv(TRAIN_DATA_PATH)
    # 폴더명만 담은 dataframe 만들기
    dir_list = df['path'].tolist()
    dir_df = pd.DataFrame(dir_list, columns = ['dir_path'])
    dir_df['label'] = dir_df['dir_path'].map(lambda x: dir_labeling(x))


    # 사람이 겹치지 않게, label 비율 유지하면서 train, valid 나눔
    train_dirs, valid_dirs = train_test_split(dir_df, test_size= 0.1, 
                                    shuffle = True, stratify= dir_df['label'], 
                                    random_state = seed)
    print('Train folders:', train_dirs.shape)
    print("Valid folders:", valid_dirs.shape)

    # 폴더에 해당하는 전체 파일 불러오기
    train_full_path = make_each_path(TRAIN_IMG_PATH, train_dirs)  # train 전체 경로 담기
    valid_full_path = make_each_path(TRAIN_IMG_PATH, valid_dirs)  # valid 전체 경로 담기
    train_paths_df = pd.DataFrame(train_full_path, columns = ['full_path'])
    valid_paths_df = pd.DataFrame(valid_full_path, columns = ['full_path'])
    train_paths_df['label'] = train_paths_df['full_path'].map(lambda x : labeling(x))
    valid_paths_df['label'] = valid_paths_df['full_path'].map(lambda x : labeling(x))
    
    # is_aug라는 column만들어 augmentation 여부 표시
    train_paths_df['is_aug'] = 0
    valid_paths_df['is_aug'] = 0
    print('Train 데이터 개수:', len(train_paths_df))
    print('Valid 데이터 개수:', len(valid_paths_df))


    ### 2. 클래스별로 augmentation 수행
    # 데이터 많은건 줄이기 -> 원본 데이터 개수가 많은 0, 1, 3, 4 클래스는 1000개만 사용
    down_label_4_df = train_paths_df[train_paths_df['label']==4].sample(n=1000, random_state=seed)
    down_label_3_df = train_paths_df[train_paths_df['label']==3].sample(n=1000, random_state=seed)
    down_label_1_df = train_paths_df[train_paths_df['label']==1].sample(n=1000, random_state=seed)
    down_label_0_df = train_paths_df[train_paths_df['label']==0].sample(n=1000, random_state=seed)

    # 각각 1000개 까지 줄인 데이터를 합침
    down_df = pd.concat([down_label_4_df, down_label_3_df, down_label_1_df, down_label_0_df])
    print('데이터 버린 클래스의 총 개수:', len(down_df))  # 1000 * 4


    # 데이터 적은건 늘리기 -> 5배/2배로 늘리는 클래스 만들고 합침, is_aug = 1 로 설정
    mul_5 = [8, 11, 14, 17]
    mul_2 = [2, 5, 6, 7, 9, 10, 12, 13, 15, 16]

    mul_5_each_df = train_paths_df[train_paths_df['label'].isin(mul_5)]
    mul_2_each_df = train_paths_df[train_paths_df['label'].isin(mul_2)]
    # print(mul_5_each_df.shape, mul_2_each_df.shape)

    mul_5_df = pd.concat([mul_5_each_df, mul_5_each_df, mul_5_each_df, mul_5_each_df, mul_5_each_df])
    mul_2_df = pd.concat([mul_2_each_df, mul_2_each_df])
    # print(mul_5_df.shape, mul_2_df.shape)

    mul_df = pd.concat([mul_5_df, mul_2_df])
    mul_df.loc[:,'is_aug'] = 1
    # print('shape:', mul_df.shape)
    print('데이터 늘린 클래스의 총 개수:', len(mul_df))  # 12488



    # 줄이거나 늘린 데이터 path를 합침
    # is_aug가 1인 녀석들은 transform을 이용해서 원본에 조금 변화를 줄것이다
    train_paths_aug_df = pd.concat([down_df, mul_df], ignore_index = True)
    train_paths_aug_df = train_paths_aug_df.sample(frac=1).reset_index(drop=True) # shuffle
    # print(train_paths_aug_df.shape)
    print('aug 이후 train 개수:', len(train_paths_aug_df))

    # 최종 train, valid dataframe 반환
    return train_paths_aug_df, valid_paths_df



# main
if __name__ == "__main__":
    print("main 시작")
    TRAIN_DATA_PATH = '/opt/ml/input/data/train/train.csv'
    TRAIN_IMG_PATH = '/opt/ml/input/data/train/images'
    train_df, valid_df = data_processing(TRAIN_DATA_PATH, TRAIN_IMG_PATH)




# is_aug 여부에 따라 transform이 달라져 transfom, aug_transform 파라미터 2개임
class MyDataset(Dataset):
    num_classes = 3 * 2 * 3

    def __init__(self, img_paths_label, transform, aug_transform = None):
        self.X = img_paths_label['full_path']
        self.y = img_paths_label['label']
        self.aug_check = img_paths_label['is_aug']
        self.transform = transform
        self.aug_transform = aug_transform

    def __getitem__(self, index):
        image = cv2.imread(self.X.iloc[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.y.iloc[index]
        
        if self.aug_check.iloc[index] == 1:
            transformed = self.aug_transform(image = image)
        else:
            transformed = self.transform(image = image)
        
        transformed_image = transformed['image']
        return transformed_image, torch.tensor(label)

    def __len__(self):
        return len(self.X)



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
