import os
import argparse
import multiprocessing
from importlib import import_module
import numpy as np

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from zmq import device
from dataset import TestDataset, MaskDataset, GenderDataset, AgeDataset


def load_mask_model(saved_model, device, num_classes=3):
    model_cls = getattr(import_module("model"), args.mask_model)
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_gender_model(saved_model, device, num_classes=2):
    model_cls = getattr(import_module("model"), args.gender_model)
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_age_model(saved_model, device, num_classes=3):
    model_cls = getattr(import_module("model"), args.age_model)
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(data_dir, mask_model_dir, gender_model_dir, age_model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mask_model = load_mask_model(mask_model_dir, device).to(device)
    gender_model = load_gender_model(gender_model_dir, device).to(device)
    age_model = load_age_model(age_model_dir, device).to(device)

    mask_model.eval()
    gender_model.eval()
    age_model.eval()
    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    dataset = TestDataset(img_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results...")

    preds = []

    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)

            _pred = mask_model(images)
            # print("mask\t", _pred[0], end='\t')
            pred = _pred.argmax(dim=-1) * 6

            _pred = gender_model(images)
            # print("gender\t", _pred[0], end='\t')
            pred += _pred.argmax(dim=-1) * 3

            _pred = age_model(images)
            # print("age\t", _pred[0], end='\t')
            pred += _pred.argmax(dim=-1)

            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--resize', type=tuple, default=(512, 384))
    parser.add_argument('--mask_model', type=str, default='ResNet18')
    parser.add_argument('--gender_model', type=str, default='ResNet18')
    parser.add_argument('--age_model', type=str, default='ResNet18')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--mask_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/wj_mask_cls'))
    parser.add_argument('--gender_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/wj_gender_cls'))
    parser.add_argument('--age_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/wj_age_cls'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    args = parser.parse_args()

    data_dir = args.data_dir

    mask_model_dir = args.mask_model_dir
    gender_model_dir = args.gender_model_dir
    age_model_dir = args.age_model_dir

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, mask_model_dir, gender_model_dir, age_model_dir, output_dir, args)