import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from transform import My_transform
from model import Multi_ModelClassification



def load_model(saved_model, num_classes, device):

    model = Multi_ModelClassification()
    path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(path, map_location=device))

    return model

def encode_multi_class(age_label, gender_label, mask_label) -> int:
    return mask_label * 6 + gender_label * 3 + age_label


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()
    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, transform=My_transform('test'))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []

    age_val_loss_items,gender_val_loss_items,mask_val_loss_items = [],[],[]
    age_val_acc_items,gender_val_acc_items,mask_val_acc_items = [],[],[]
    figure = None
    for val_batch in loader:
        inputs=val_batch
        inputs = inputs.to(device)

        age_outs, gender_outs, mask_outs = model(inputs)
        age_preds = torch.argmax(age_outs, dim=-1)
        gender_preds = torch.argmax(gender_outs, dim=-1)
        mask_preds = torch.argmax(mask_outs, dim=-1)
        for age,gender,mask in zip(age_preds.cpu().numpy(),gender_preds.cpu().numpy(),mask_preds.cpu().numpy()):
            pred = dict_age[str(age)]+'|'+dict_gender[str(gender)]+'|'+dict_mask[str(mask)]    
            preds.append(encode_multi_class(age,gender,mask))          
            

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'ttt.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/code/baseline/v2/model/exp10'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/code/baseline/v2/output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
