import argparse
from cProfile import label
import os
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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import wandb
from tqdm import tqdm, tqdm_notebook
from albumentations.pytorch import transforms
import albumentations as A

from model import Resnet18
from dataset import MyDataset, data_processing


# -- settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
SAVE_DIR = './result/checkpoint/resnet18'

# -- hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LOG_INTERVAL = 20
LR_DECAY_STEP = 5


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
# wandb.init(project="mask_classification", entity="hannayeoniee", name="test_0226")
save_name = "test_ny_Resnet18_Adam_CEloss_batch64_aug-profile"
wandb.init(project="mask_classification", entity="level1-nlp-07", name=save_name)


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
scheduler = StepLR(optimizer, LR_DECAY_STEP, gamma=0.5)  # step size마다 gamma비율로 lr 감소시킴
dataloaders = {"train" : train_dataloader,
              "test" : valid_dataloader}


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


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print('LR: {:0.6f} Batch: {}'.format(optimizer.param_groups[0]['lr'],BATCH_SIZE))
#     for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataset)):
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        inputs= inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
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
    if epoch%5==0:
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': train_loss/(batch_idx+1),
          "acc": 100*correct/total
          }, f"./result/checkpoint/resnet18_epoch_{epoch}_loss_{0:0.5f}.pt".format(loss))
    
    # wandb logging
    wandb.log({'Epoch': epoch,
               'Train loss': train_loss/(batch_idx+1),
               'Train accuracy': 100*correct/total})

    
        
def test(epoch):
    global acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # wandb에 기록할 테스트 이미지들
    valid_images = []

    with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(tqdm(valid_dataset)):
        for batch_idx, (inputs, targets) in enumerate(tqdm(valid_dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
#             print('예측결과: ', f'Predicted: {predicted[0].item()}, Ground-truth: {targets[0]}')
            valid_images.append(wandb.Image(inputs[0],
                                            caption=f'Predicted: {predicted[0].item()}, Ground-truth: {targets[0]}'))

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
        if acc<=100.*correct/total:
            acc = 100.*correct/total
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': test_loss/(batch_idx+1),
              "acc": 100*correct/total
              }, f"./result/checkpoint/resnet18_best_performance_acc_{0:0.5f}_loss_{0:0.5f}.pt".format(acc, loss))
#             print("acc: {0:0.5f}, loss: {0:0.5f}".format(acc, loss))

            
        # wandb logging
        wandb.log({'Epoch': epoch,
                   'Test loss': test_loss/(batch_idx+1),
                   'Test accuracy': 100*correct/total})
        


# 학습할 때 batch마다 optimizer.step(), epoch마다 scheduler.step() 수행
def train2(epoch):
    # train loop
    model.train()
    loss_value = 0
    matches = 0
    for idx, train_batch in enumerate(tqdm(train_dataloader)):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=-1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        matches += (preds == labels).sum().item()

    
        if (idx + 1) % LOG_INTERVAL == 0:
            train_loss = loss_value / LOG_INTERVAL
            train_acc = matches / BATCH_SIZE / LOG_INTERVAL
            current_lr = get_lr(optimizer)
            print(
                f"Epoch[{epoch}/{EPOCHS}]({idx + 1}/{len(train_dataloader)}) || "
                f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
            )
            # # tensorboard logging
            # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
            # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
            # wandb logging
            wandb.log({'Epoch': epoch, 
                    'Train loss': train_loss, 
                    'Train accuracy': train_acc})
            
            loss_value = 0
            matches = 0

    # scheduler.step()  # learning rate scheduler



def test2(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    # best_val_acc = 0
    # best_val_loss = np.inf

    # for epoch in range(EPOCHS):

    valid_images = []  # wandb에 기록할 테스트 이미지들

    # val loop
    with torch.no_grad():
        print("Calculating validation results...")
        # model.eval()
        val_loss_items = []
        val_acc_items = []
        # figure = None

        for val_batch in valid_dataloader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)  # get the index of the max log-probability
            val_loss += F.nll_loss(outs, labels, reduction='sum').item()  # sum up batch loss
            correct += preds.eq(labels.view_as(preds)).sum().item()
            # wandb에 현재 배치에 포함된 첫 번째 이미지에 대한 추론 결과 기록 
            valid_images.append(wandb.Image(inputs[0], 
                                            caption=f'Predicted: {preds[0].item()}, Ground-truth: {labels[0]}'))


    val_loss /= len(valid_dataset)
    val_acc = 100. * correct / len(valid_dataset)

    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(valid_dataset),
        100. * correct / len(valid_dataset)))
    
    # wandb logging
    wandb.log({'Epoch': epoch, 
                'Valid loss': val_loss,
                'Valid accuracy': val_acc,
                'Valid images': valid_images})




def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train3(epoch):
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

    # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #     % (train_loss/(idx+1), 100.*correct/total, correct, total))
    final_train_loss = train_loss/(idx+1)
    final_train_acc = 100*correct/total

    current_lr = get_lr(optimizer)
    print(
        f"Epoch[{epoch}/{EPOCHS}]({idx + 1}/{len(train_dataloader)}) || Batch {BATCH_SIZE} || "
        f"Train loss {0:0.5f} || Train accuracy {0:0.5f} ({correct}/{total}) || lr {current_lr}".format(final_train_loss, final_train_acc)
        )

    # model save
    if epoch % 2 == 0:
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': final_train_loss,
          'acc': final_train_acc
          }, f"./result/checkpoint/resnet18_epoch_{epoch}_loss_{0:0.5f}.pt".format(final_train_loss))
    
    # wandb logging
    wandb.log({'Epoch': epoch,
                'Train accuracy': final_train_acc,
                'Train loss': final_train_loss})


best_acc = 0.0
def test3(epoch):
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
        print(f"Val loss {final_val_loss:0.5f} || Val accuracy {final_val_acc:0.5f} ({correct}/{total})")

        if best_acc <= final_val_acc:
            best_acc = final_val_acc
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': final_val_loss,
              "acc": final_val_acc
              }, f"./result/checkpoint/resnet18_best_performance_acc_{0:0.5f}_loss_{0:0.5f}.pt".format(best_acc, final_val_loss))

        # wandb logging
        wandb.log({'Epoch': epoch, 
                    'Valid loss': final_val_loss,
                    'Valid accuracy': final_val_acc,
                    'Valid images': valid_images})



def main():
    for epoch in range(EPOCHS):
        train3(epoch)
        test3(epoch)
        scheduler.step()


if __name__ == "__main__":
    main()







# 학습할 때 batch마다 optimizer.step(), epoch마다 scheduler.step() 수행
def train_test():
    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(EPOCHS):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(tqdm(train_dataloader)):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()

        
            if (idx + 1) % LOG_INTERVAL == 0:
                train_loss = loss_value / LOG_INTERVAL
                train_acc = matches / BATCH_SIZE / LOG_INTERVAL
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{EPOCHS}]({idx + 1}/{len(train_dataloader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                # # tensorboard logging
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                # wandb logging
                wandb.log({'Epoch': epoch, 
                        'Train loss': train_loss, 
                        'Train accuracy': train_acc})
                
                loss_value = 0
                matches = 0

        scheduler.step()  # learning rate scheduler
        # wandb에 기록할 테스트 이미지들
        valid_images = []

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in valid_dataloader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                # wandb에 현재 배치에 포함된 첫 번째 이미지에 대한 추론 결과 기록 
                valid_images.append(wandb.Image(inputs[0], 
                                                caption=f'Predicted: {preds[0].item()}, Ground-truth: {labels[0]}'))
                # valid_images.append(wandb.Image(inputs, 
                #                                 caption=f'Predicted: {preds.item()}, Ground-truth: {labels}'))


            val_loss = np.sum(val_loss_items) / len(valid_dataloader)
            val_acc = np.sum(val_acc_items) / len(valid_dataset)
            best_val_loss = min(best_val_loss, val_loss)

            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                # torch.save(model.module.state_dict(), f"{SAVE_DIR}/best.pth")
                # torch.save(net, f"{model_saved}/loss_resnet18_CEloss_batchsize:{BATCH_SIZE}_lr:{LEARNING_RATE}.pt")
                # torch.save(model.module.state_dict(), f"{SAVE_DIR}_best.pt")
                torch.save(model, f"{SAVE_DIR}_best.pt")

                best_val_acc = val_acc

            # torch.save(model.module.state_dict(), f"{SAVE_DIR}_last.pt")
            torch.save(model, f"{SAVE_DIR}_last.pt")
            print(f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}")
            
            # wandb logging
            wandb.log({'Epoch': epoch, 
                       'Valid loss': val_loss, 
                       'Valid accuracy': val_acc,
                       'Valid images': valid_images})
            print()



# # 민준님 버전
# best_test_accuracy = 0.
# best_test_loss = 9999.
# model_saved = "./result"

# for epoch in range(EPOCHS):
#     for phase in ["train", "test"]:
#         running_loss = 0.
#         running_acc = 0.
#         if phase == "train":
#             net.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
#         elif phase == "test":
#             net.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함
        
#         for ind, (images, labels) in enumerate(dataloaders[phase]):
#             images = images.to(device)
#             labels = labels.to(device)
        
#             optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

#             with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
#                 logits = net(images)
#                 _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함  
#                 loss = criterion(logits, labels)

#                 if phase == "train":
#                     loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
#                     optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

#             running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
#             running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장

#         # 한 epoch이 모두 종료되었을 때,
#         epoch_loss = running_loss / len(dataloaders[phase].dataset)
#         epoch_acc = running_acc / len(dataloaders[phase].dataset)

#         print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
#         if phase == "test" and best_test_accuracy < epoch_acc: # phase가 test일 때, best accuracy 계산
#             best_test_accuracy = epoch_acc
#             os.makedirs(model_saved, exist_ok=True)
#             torch.save(net, f"{model_saved}/accr_resnet18_CEloss_batchsize:{BATCH_SIZE}_lr:{LEARNING_RATE}.pt")
#         if phase == "test" and best_test_loss > epoch_loss: # phase가 test일 때, best loss 계산
#             best_test_loss = epoch_loss
#             os.makedirs(model_saved, exist_ok=True)
#             torch.save(net, f"{model_saved}/loss_resnet18_CEloss_batchsize:{BATCH_SIZE}_lr:{LEARNING_RATE}.pt")
            
# print("학습 종료!")
# print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}")
