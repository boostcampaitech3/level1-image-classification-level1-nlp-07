import argparse
from cProfile import label
from email.policy import default
import os
import errno
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import wandb
from tqdm import tqdm, tqdm_notebook
from albumentations.pytorch import transforms
import albumentations as A

from model import Resnet18
from dataset import MyDataset, data_processing


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer, epoch, LR):
    if epoch <= 10:
        lr = LR     
    elif epoch <= 15:
        lr = LR * 0.5           
    else:
        lr = LR * 0.1   
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 폴더 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx, train_batch in enumerate(tqdm(train_dataloader)):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()


    final_train_loss = train_loss/(idx+1)
    final_train_acc = 100*correct/total

    current_lr = get_lr(optimizer)
    print(
        f"Epoch[{epoch}/{EPOCHS}]({idx + 1}/{len(train_dataloader)}) || Batch {BATCH_SIZE} || "
        f"Train loss {final_train_loss:.3f} || Train accuracy {final_train_acc:.3f} ({correct}/{total}) || lr {current_lr}"
        )

    # model save
    if epoch % 5 == 0:
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': final_train_loss,
          'acc': final_train_acc
        #   }, f"./result/checkpoint/resnet18_epoch_{epoch}_loss_{final_train_loss:.3f}.pt")
          }, f"{SAVE_PATH}/epoch_{epoch}_loss_{final_train_loss:.3f}.pt")
    
    # wandb logging
    wandb.log({'Epoch': epoch,
                'Train accuracy': final_train_acc,
                'Train loss': final_train_loss,
                'Learning rate': current_lr})


best_acc = 0.0
def test(epoch):
    global best_acc
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # wandb에 기록할 테스트 이미지들
    valid_images = [] 

    with torch.no_grad():
        print("Calculating validation results...")

        for idx, val_batch in enumerate(tqdm(valid_dataloader)):
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)  # get the index of the max log-probability
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            # wandb에 현재 배치에 포함된 첫 번째 이미지에 대한 추론 결과 기록 
            valid_images.append(wandb.Image(inputs[0], 
                                            caption=f'Predicted: {preds[0].item()}, Ground-truth: {labels[0]}'))

        # 한 epoch이 모두 종료되었을 때
        final_val_loss = val_loss/(idx+1)
        final_val_acc = 100*correct/total
        print(f"Val loss {final_val_loss:0.3f} || Val accuracy {final_val_acc:0.3f} ({correct}/{total})")

        if best_acc <= final_val_acc:
            best_acc = final_val_acc
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': final_val_loss,
              "acc": final_val_acc
            #   }, f"./result/checkpoint/resnet18_best_performance_acc_{best_acc:.3f}_loss_{final_val_loss:.3f}.pt")
              }, f"{SAVE_PATH}/best_performance_acc_{best_acc:.3f}_loss_{final_val_loss:.3f}.pt")

        # wandb logging
        wandb.log({'Epoch': epoch, 
                    'Valid loss': final_val_loss,
                    'Valid accuracy': final_val_acc,
                    'Valid images': valid_images})


if __name__ == "__main__":
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4


    # -- load data
    # 데이터셋 경로 지정
    TRAIN_DATA_PATH = '/opt/ml/input/data/train/train.csv'
    TRAIN_IMG_PATH = '/opt/ml/input/data/train/images'


    # 데이터 전처리 & 증강 및 버리기 
    train_paths_aug_df, valid_paths_df = data_processing(TRAIN_DATA_PATH, TRAIN_IMG_PATH)
    print("Train data 개수:",len(train_paths_aug_df))
    print("Valid data 개수:",len(valid_paths_df))


    # -- augmentation
    # 일반용 transform
    transform = A.Compose([
        A.Resize(512, 384),
        A.Normalize(mean=0.5, std=0.2),
        # A.Normalize(mean = mean, std = std),
        transforms.ToTensorV2(transpose_mask=True)
        #A.CenterCrop(height=300, width=300, p=1)
    ])

    # aug용 transform
    aug_transform = A.Compose([
        A.Resize(512, 384),
        A.HorizontalFlip(p = 0.5), # 좌우 대칭, p는 확률로 1이면 항상 하는것
        A.Rotate(limit=7, p = 0.5), # 회전, 최고를 7도로 줌
        A.RandomBrightnessContrast(0.1, p = 0.5), # 밝기?
        A.RandomGamma(gamma_limit=(80,120), p = 0.5), # 잘 모르겠지만 밝기좀 변함, limit 잘 조정해야한다
        A.Normalize(mean=0.5, std=0.2), # 정규화, 이상하게 albutation은 텐서로 바꾸기전에 이미지에 정규화 때린다.
        # A.Normalize(mean = mean, std = std),
        transforms.ToTensorV2(transpose_mask=True) # tensor로 변형, transpose_mask 해야 C, H W로 나옴
        #A.CenterCrop(height=300, width=300, p=1) # crop, 나중에 시도해봐야지
    ])
    print('==> Preparing data..')


    ## -- model
    model = Resnet18(num_classes=18)
    model = model.to(device)


    # naming convention: model_optimizer_batch_lr_kfold_data-aug
    SAVE_DIR = './result/checkpoint/'
    # SAVE_NAME = "ny_Resnet18_Adam_CEloss_batch64_aug-profile_steplr"
    # @@@@@@@@@@@@@@@@ 본인 name으로 바꾸주세요!!! @@@@@@@@@@@@@@@@
    SAVE_NAME = "(test)_model_optimizer_batch_lr_kfold_data-aug"
    SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)
    createFolder(SAVE_PATH)
    # wandb.init(project="mask_classification", entity="hannayeoniee", name=SAVE_NAME)
    wandb.init(project="mask_classification", entity="level1-nlp-07", name=SAVE_NAME)


    ## -- data_loader
    train_dataset = MyDataset(train_paths_aug_df, transform, aug_transform)
    train_dataloader = DataLoader(train_dataset,
                                batch_size = BATCH_SIZE, 
                                shuffle = True)
    valid_dataset = MyDataset(valid_paths_df, transform)
    valid_dataloader = DataLoader(valid_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = True)

    # -- loss & metric
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)  # step size마다 gamma를 곱해 lr 감소시킴

    dataloaders = {"train" : train_dataloader,
                    "test" : valid_dataloader}


    # train & test 진행
    for epoch in range(EPOCHS):
        train(epoch)
        test(epoch)
        scheduler.step()
