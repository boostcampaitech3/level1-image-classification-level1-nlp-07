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

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

test_dir = "/opt/ml/input/data/eval/"

# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

dataset = TestDataset(image_paths, transform)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

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
checkpoint = torch.load('/opt/ml/result/checkpoint/mask/mask_resnet18_best_performance_99.84126984126983_1.0889752957154997e-05.pt')
net_mask.load_state_dict(checkpoint['model_state_dict'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_mask = net_mask.to(device)
net_mask.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
# 3 2 3
dict_label={'011':0,'111':1,'211':2,'001':3,'101':4,'201':5,'010':6,'110':7,'210':8,'000':9,'100':10,'200':11,'012':12,'112':13,'212':14,'002':15,'102':16,'202':17}
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred_age = net_age(images)
        pred_age = pred_age.argmax(dim=-1)
        pred_gender = net_gender(images)
        pred_gender = pred_gender.argmax(dim=-1)
        pred_mask = net_mask(images)
        pred_mask = pred_mask.argmax(dim=-1)
        for age,gender,mask in zip(pred_age.cpu().numpy(),pred_gender.cpu().numpy(),pred_mask.cpu().numpy()):
            pred = str(age)+str(gender)+str(mask)
            all_predictions.append(dict_label[pred])
        
print(all_predictions)
        
submission['ans'] = all_predictions

# # 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')