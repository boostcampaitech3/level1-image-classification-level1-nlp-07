import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from sklearn.model_selection import train_test_split

import wandb
from tqdm import tqdm, tqdm_notebook
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from model import MyNet
from dataset import GenderDataset, MaskDataset, AgeDataset

# 테스트 데이터셋 폴더 경로를 지정해주세요.
train_dir = '/opt/ml/input/data/train'
test_dir = '/opt/ml/input/data/eval'

# meta 데이터와 이미지 경로를 불러옵니다.
train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'))
image_dir = os.path.join(train_dir, 'images')

image_paths = [os.path.join(image_dir, img_id) for img_id in train_df.path]

img_list=[]
for i in image_paths:
    for j in os.listdir(i):
        if j[0]!='.':
            img_list.append(i+'/'+j)
# print(img_list)
            
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.labeling(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
            
        return image,label,self.img_paths[index]

    def __len__(self):
        return len(self.img_paths)
    
    def labeling(self,name):
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

transform_test = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])


print('==> Preparing data..')

BATCH_SIZE = 64

# train, test = train_test_split(img_list, test_size=0.1, shuffle=True, random_state=1004)
_, test = train_test_split(img_list, test_size=0.1, shuffle=True, random_state=1004)
print("test 개수:",len(test))

test_dataset = TestDataset(test, transform_test)
test_set =  DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)

# age
net_age = MyNet(3)
checkpoint = torch.load('/opt/ml/result/checkpoint/age/age_resnet18_best_performance_99.94708994708995_6.065589695936069e-05.pt')
net_age.load_state_dict(checkpoint['model_state_dict'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_age = net_age.to(device)
net_age.eval()

# gender
net_gender = MyNet(2)
checkpoint = torch.load('/opt/ml/result/checkpoint/gender/gender_resnet18_best_performance_100.0_1.0082588232762646e-05.pt')
net_gender.load_state_dict(checkpoint['model_state_dict'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_gender = net_gender.to(device)
net_gender.eval()

# mask
net_mask = MyNet(3)
checkpoint = torch.load('/opt/ml/result/checkpoint/mask_resnet18_best_performance_99.84126984126983_0.0013221995905041695.pt')
net_mask.load_state_dict(checkpoint['model_state_dict'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_mask = net_mask.to(device)
net_mask.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
# 3 2 3
dict_label={'011':0,'111':1,'211':2,'001':3,'101':4,'201':5,'010':6,'110':7,'210':8,'000':9,'100':10,'200':11,'012':12,'112':13,'212':14,'002':15,'102':16,'202':17}
result = pd.DataFrame({'label':[],
                             'pred':[],'data':[],'image_path':[]})
false_sample=[]
for images,labels,path in test_set:
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        pred_age = net_age(images)
        pred_age = pred_age.argmax(dim=-1)
        pred_gender = net_gender(images)
        pred_gender = pred_gender.argmax(dim=-1)
        pred_mask = net_mask(images)
        pred_mask = pred_mask.argmax(dim=-1)
        for age,gender,mask,labels,path in zip(pred_age.cpu().numpy(),pred_gender.cpu().numpy(),pred_mask.cpu().numpy(),labels.cpu().numpy(),path):
            
            pred = str(age)+str(gender)+str(mask)
            data_to_insert = {'label': labels, 'pred': dict_label[pred],'data':pred,'image_path':path}
            result = result.append(data_to_insert, ignore_index=True)
            if int(dict_label[pred])!=int(labels):
                false_sample.append(path)

print(false_sample)

# # 제출할 파일을 저장합니다.
result.to_csv(os.path.join('./', 'test.csv'), index=False)
print('test inference is done!')