import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset,MaskSplitByProfileDataset
from loss import create_criterion
from model import Multi_ModelClassification
from transform import My_transform
import wandb


# wandb.init(project="mask_classification", entity='hyeyoon',name='strong-aug')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

    
def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    os.makedirs(save_dir, exist_ok=True)
    print('model save path',save_dir)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir, transform=My_transform('train')
    )

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    print('train data 개수:',len(train_set))
    print('val data 개수:',len(val_set))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model = Multi_ModelClassification().to(device)
    model = torch.nn.DataParallel(model)
#     model = Multi_ModelClassification()
#     path = os.path.join('/opt/ml/code/baseline/v2/model/exp4', 'last.pth')
#     model.load_state_dict(torch.load(path, map_location=device))
#     model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    print(optimizer)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    age_best_val_acc,gender_best_val_acc,mask_best_val_acc=0,0,0
    age_best_val_loss,gender_best_val_loss,mask_best_val_loss=np.inf,np.inf,np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        age_loss_value,gender_loss_value,mask_loss_value=0,0,0
        age_matches,gender_matches,mask_matches = 0,0,0
        for idx, train_batch in enumerate(train_loader):
            inputs,age_label,gender_label,mask_label=train_batch
            inputs = inputs.to(device)
            age_label = age_label.to(device)
            gender_label = gender_label.to(device)
            mask_label = mask_label.to(device)
            
            optimizer.zero_grad()

            age_outs, gender_outs, mask_outs = model(inputs)
            
            age_preds = torch.argmax(age_outs, dim=-1)
            age_loss = criterion(age_outs, age_label)
            
            gender_preds = torch.argmax(gender_outs, dim=-1)
            gender_loss = criterion(gender_outs, gender_label)
            
            mask_preds = torch.argmax(mask_outs, dim=-1)
            mask_loss = criterion(mask_outs, mask_label)
            # loss balancing (이렇게 주는게 맞나)
            loss = 0.5*age_loss+0.25*gender_loss+0.25*mask_loss

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
           
            age_loss_value+=age_loss.item()
            gender_loss_value+=gender_loss.item()
            mask_loss_value+=mask_loss.item()
            
            age_matches += (age_preds == age_label).sum().item()
            gender_matches += (gender_preds == gender_label).sum().item()
            mask_matches += (mask_preds == mask_label).sum().item()
            if (idx + 1) % args.log_interval == 0:
                age_train_loss = age_loss_value / args.log_interval
                age_train_acc = age_matches / args.batch_size / args.log_interval
                gender_train_loss = gender_loss_value / args.log_interval
                gender_train_acc = gender_matches / args.batch_size / args.log_interval
                mask_train_loss = mask_loss_value / args.log_interval
                mask_train_acc = mask_matches / args.batch_size / args.log_interval
                total_loss = loss_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"age training loss {age_train_loss:4.4} || training accuracy {age_train_acc:4.2%} || lr {current_lr}"
                )
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"gender training loss {gender_train_loss:4.4} || training accuracy {gender_train_acc:4.2%} || lr {current_lr}"
                )
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"mask training loss {mask_train_loss:4.4} || training accuracy {mask_train_acc:4.2%} || lr {current_lr}"
                )

                loss_value = 0
                age_matches,gender_matches,mask_matches = 0,0,0

        scheduler.step()
        wandb.log({'total_loss': total_loss, 'epoch': epoch})
        wandb.log({'age_train_accuracy': age_train_acc, 'age_train_loss': age_train_loss,'epoch': epoch})
        wandb.log({'gender_train_accuracy': gender_train_acc, 'gender_train_loss': gender_train_loss,'epoch': epoch})
        wandb.log({'mask_train_accuracy': mask_train_acc, 'mask_train_loss': mask_train_loss,'epoch': epoch})

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            age_val_loss_items,gender_val_loss_items,mask_val_loss_items = [],[],[]
            age_val_acc_items,gender_val_acc_items,mask_val_acc_items = [],[],[]
            figure = None
            for val_batch in val_loader:
                inputs,age_label,gender_label,mask_label=val_batch
                inputs = inputs.to(device)
                age_label = age_label.to(device)
                gender_label = gender_label.to(device)
                mask_label = mask_label.to(device)
                
                age_outs, gender_outs, mask_outs = model(inputs)
                age_preds = torch.argmax(age_outs, dim=-1)
                age_loss = criterion(age_outs, age_label).item()
                gender_preds = torch.argmax(gender_outs, dim=-1)
                gender_loss = criterion(gender_outs, gender_label).item()
                mask_preds = torch.argmax(mask_outs, dim=-1)
                mask_loss = criterion(mask_outs, mask_label).item()
                
                age_matches = (age_preds == age_label).sum().item()
                gender_matches = (gender_preds == gender_label).sum().item()
                mask_matches = (mask_preds == mask_label).sum().item()


                age_val_loss_items.append(age_loss)
                age_val_acc_items.append(age_matches)
                gender_val_loss_items.append(gender_loss)
                gender_val_acc_items.append(gender_matches)
                mask_val_loss_items.append(mask_loss)
                mask_val_acc_items.append(mask_matches)

            age_val_loss = np.sum(age_val_loss_items) / len(val_loader)
            age_val_acc = np.sum(age_val_acc_items) / len(val_set)
            gender_val_loss = np.sum(gender_val_loss_items) / len(val_loader)
            gender_val_acc = np.sum(gender_val_acc_items) / len(val_set)
            mask_val_loss = np.sum(mask_val_loss_items) / len(val_loader)
            mask_val_acc = np.sum(mask_val_acc_items) / len(val_set)
            
            age_best_val_loss = min(age_best_val_loss, age_val_loss)
            gender_best_val_loss = min(gender_best_val_loss, gender_val_loss)
            mask_best_val_loss = min(mask_best_val_loss, mask_val_loss)

            if age_val_acc > age_best_val_acc:
                print(f"New best model for val accuracy : {age_val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                age_best_val_acc = age_val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] age acc : {age_val_acc:4.2%}, loss: {age_val_loss:4.2} || "
                f"best acc : {age_best_val_acc:4.2%}, best loss: {age_best_val_loss:4.2}"
            )
            print(
                f"[Val] gender acc : {gender_val_acc:4.2%}, loss: {gender_val_loss:4.2} || "
                f"best acc : {gender_best_val_acc:4.2%}, best loss: {gender_best_val_loss:4.2}"
            )
            print(
                f"[Val] mask acc : {mask_val_acc:4.2%}, loss: {mask_val_loss:4.2} || "
                f"best acc : {mask_best_val_acc:4.2%}, best loss: {mask_best_val_loss:4.2}"
            )

            wandb.log({'total_loss': total_loss, 'epoch': epoch})
            wandb.log({'age_val_accuracy': age_val_acc, 'age_val_loss': age_val_loss,'epoch': epoch})
            wandb.log({'gender_val_accuracy': gender_val_acc, 'gender_val_loss': gender_val_loss,'epoch': epoch})
            wandb.log({'mask_val_accuracy': mask_val_acc, 'mask_val_loss': mask_val_loss,'epoch': epoch})
            
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/code/baseline/v2/model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
