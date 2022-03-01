
import pandas as pd 
import os
import albumentations as A
import cv2
from PIL import Image
import numpy as np
from dataset import TestDataset
import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb
from tqdm import tqdm, tqdm_notebook
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from collections import deque

# path settings
train_dir =  '/opt/ml/input/data/train'
# augmentation 이미지 저장할 경로
AUG_DIR = '/opt/ml/code/augtest/'
# train.csv 파일 경로
TRAIN_CSV = '/opt/ml/input/data/train/train.csv'

train_df = pd.read_csv(TRAIN_CSV) 
train_df = train_df

image_dir = os.path.join(train_dir, 'images')

image_paths = [os.path.join(image_dir, img_id) for img_id in train_df.path]

image_paths = deque(image_paths)

img_list=deque()
while image_paths:
    i =image_paths.popleft()
    for j in os.listdir(i):
        if j[0]!='.':
            img_list.append(i+'/'+j)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])

test_dataset = TestDataset(img_list,transform_test)
test_set =  DataLoader(test_dataset, batch_size=1,shuffle=False)

img_list_x4=deque()
img_list_x3=deque()
for images,labels,path in test_set:
    for cls,pth in zip(labels.cpu().numpy(),path):
        # augmentation 진행할 class 설정
        if cls in [8,11,14,17]:  
            img_list_x4.append(pth)
        if cls not in [0,1,3,4]:
            img_list_x3.append(pth)
            
print(len(img_list),img_list[0])       
            
# # https://albumentations.ai/docs/examples/example/
def augmentation_func(img_list,name_start=''):
    
    # augmentation 수행하여 이미지만 폴더에 저장
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.OneOf([
              A.RandomBrightnessContrast(p=1),
              A.GaussNoise(p=1),], p=0.8)])
#     transform = A.Compose([
#             A.HorizontalFlip(p=0.6),
#             A.Rotate(limit=10,p=0.3),
#             A.OneOf([
#                 A.IAAAdditiveGaussianNoise(),
#                 A.GaussNoise(),
#             ], p=0.3),
#             A.OneOf([
#                 A.MotionBlur(p=1.0), 
#                 A.MedianBlur(blur_limit=5, p=1),  
#                 A.Blur(blur_limit=3, p=1), 
#             ], p=0.3),
#             A.OneOf([
#                 A.CLAHE(clip_limit=2),
#                 A.IAASharpen(p=1),
#                 A.IAAEmboss(p=1),
#                 A.RandomBrightnessContrast(p=1),            
#             ], p=0.4),
#             A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
#         ])
    
    # 저장할 경로
    img_dir = AUG_DIR
    print(len(img_list))
    for img in img_list:
        info,mask = img.split('/')[-2:]
        pillow_image = Image.open(img)
        image = np.array(pillow_image)
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_dir+name_start+info+'-'+mask, image)
    
for i in range(1,8):
    if 1<=i<=3:
        augmentation_func(img_list_x3,str(i))
    else:
        augmentation_func(img_list_x4,str(i))

# # 포맷 맞춰서 train.csv에 저장/이미지 폴더 저장 (사람 id 10000부터 카운팅하기, 겹치지 않게)
arr = []

for i in os.listdir(AUG_DIR):
    if i[0]!='.':
        arr.append(i)
        
arr = sorted(arr)

id={}
train_data_path = '/opt/ml/input/data/train/images/'
# train_data_path = '/opt/ml/code/sample/'
for img in arr:
    ''' data format
    003748_male_Asian_50 (사람 폴더)
    -incorrect_mask.jpg
    -mask1.jpg
    -mask2.jpg
    -mask3.jpg
    -mask4.jpg
    -mask5.jpg
    -normal.jpg
    '''
    # img format: 003528-1_male_Asian_60-incorrect_mask.jpg
    
    # 폴더이름
    img_id="-".join(img.split('-')[:-1])
    # 이미지 이름
    mask_cls = img.split('-')[-1]

    # 폴더 만들기 (사람 id별 폴더)
    os.makedirs(train_data_path+img_id, exist_ok=True)
            
    pillow_image = Image.open(AUG_DIR+img)
    image = np.array(pillow_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(train_data_path+img_id+'/'+mask_cls, image)
        
    if img_id not in id.keys():
        id[img_id]=1

        new_data = {
            'id' : [None],
            'gender' : [None],
            'race' : [None],
            'age': [None],
            'path': [img_id]
        }
        new_df = pd.DataFrame(new_data)
        train_df = pd.concat([train_df,new_df],ignore_index=False)
  
train_df.to_csv(TRAIN_CSV)