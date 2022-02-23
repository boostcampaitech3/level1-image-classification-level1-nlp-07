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

net = MyNet(num_classes=2)
print(net)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

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

transform_train = transforms.Compose([
                                # transforms.CenterCrop(32),
                                # transforms.RandomCrop(32, padding=4),
                                # transforms.RandomHorizontalFlip(),
                                transforms.Resize((512, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

transform_test = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])


print('==> Preparing data..')

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

# train, test = train_test_split(img_list, test_size=0.1, shuffle=True, random_state=1004)
train, test = train_test_split(img_list, test_size=0.1, shuffle=True, random_state=1004)
print("test 개수:",len(test))
print("train 개수:",len(train))

train_dataset = GenderDataset(train, transform_train, train=True)
train_set = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)

test_dataset = GenderDataset(test, transform_test, train=True)
test_set =  DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)


config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate" : LEARNING_RATE}

wandb.init(project="mask_classification", entity='hyeyoon')

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
# optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, epoch, LR):
    if epoch <= 10:
        lr = LR     
    elif epoch <= 15:
        lr = LR * 0.5           
    else:
        lr = LR * 0.1   
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print('LR: {:0.6f} Batch: {}'.format(optimizer.param_groups[0]['lr'],BATCH_SIZE))
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_set)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
            
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # model save
#     if epoch%5==0:
#         torch.save({
#           'epoch': epoch,
#           'model_state_dict': net.state_dict(),
#           'optimizer_state_dict': optimizer.state_dict(),
#           'loss': train_loss/(batch_idx+1),
#           "acc": 100*correct/total
#           }, f"/opt/ml/result/checkpoint/gender_resnet18_e_{epoch}_{loss}.pt")
      
    wandb.log({'train_accuracy': 100*correct/total, 'train_loss': train_loss/(batch_idx+1)})

acc = 0.0
        
def test(epoch):
    global acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_set)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if acc<=100.*correct/total:
            acc = 100.*correct/total
            torch.save({
              'epoch': epoch,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': test_loss/(batch_idx+1),
              "acc": 100*correct/total
              }, f"/opt/ml/result/checkpoint/gender_resnet18_best_performance_{acc}_{loss}.pt")
        
        wandb.log({'test_accuracy': 100*correct/total, 'test_loss': test_loss/(batch_idx+1)})
   
start_epoch = 1
for epoch in range(start_epoch, start_epoch+EPOCHS):
    adjust_learning_rate(optimizer, epoch, LR=LEARNING_RATE)
    
    train(epoch)
    test(epoch)