import argparse
import glob
from importlib import import_module
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from lr_scheduler import CosineAnnealingWarmUpRestarts
from torch.utils.tensorboard import SummaryWriter

import wandb
from loss import create_criterion

warnings.filterwarnings(action='ignore')


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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("이 기기로 학습합니다:", device)

    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir
    )

    num_classes = 3 # wear, incorrect, not wear

    transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = transform_module()
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True
    )

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=num_classes).to(device)
    # print(model)

    model = torch.nn.DataParallel(model)

    # wandb.init(project="mask_classification", entity="level1-nlp-07", name="wj_mask_cls")
    wandb.init(project="mask_classification", entity="level1-nlp-07", name="test_wj_mask_cls_epoch10_yolo_focal")
    wandb.config.update(args)

    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )


    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer=optimizer, 
        T_0=150,
        T_mult=1,
        eta_max=0.1,
        T_up=10,
        gamma=0.5
    )

    if type(scheduler).__name__ == "CosineAnnealingWarmUpRestarts":
        print("CosineAnnealingWarmUpRestarts scheduler should set the learning rate to a value close to 0!")
        for g in optimizer.param_groups:
            g['lr'] = 0
        print("Learning rate changed to 0")


    logger = SummaryWriter(log_dir=save_dir)

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf

    for epoch in range(args.epochs):
        model.train()
        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs['image'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)

            preds = torch.argmax(outs, dim=-1)

            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            matches += (preds == labels).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                wandb.log({'Epoch': epoch, 'train_loss': train_loss, 'train_accuracy':train_acc, 'lr':current_lr})

                loss_value = 0
                matches = 0

            scheduler.step()

        with torch.no_grad():
            print("Calculation validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None

            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs['image'].type(torch.FloatTensor).to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc: {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best_loss: {best_val_loss:4.2}"
            )

            wandb.log({'Epoch': epoch, 'valid_loss': val_loss, 'valid_accuracy':val_acc})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dataset', type=str, default="MaskDataset")
    parser.add_argument('--augmentation', type=str, default="BaseAugmentation")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--valid_batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default="ResNet18")
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--criterion', type=str, default="cross_entropy")
    parser.add_argument('--lr_decay_step', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--name', default="wj_mask_cls")

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
