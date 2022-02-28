import os
import cv2
from torch.utils.data import Dataset, Subset
import random
from albumentations.pytorch import transforms
import albumentations as A

class NotAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(512, 384), 
            transforms.ToTensorV2(transpose_mask=True)
        ])

    def __call__(self, image):
        return self.transform(image=image)

class BaseAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(512, 384), 
            A.Normalize(mean=0.5, std=0.2),
            transforms.ToTensorV2(transpose_mask=True)
        ])

    def __call__(self, image):
        return self.transform(image=image)


class MaskClassificationDataset(Dataset):
    def __init__(self, data_dir='/opt/ml/input/data/train/images/', val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.val_ratio = val_ratio
        
        self.indices = {"train":[], "val":[]}
        self.img_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        
        self.setup() # img_paths, labels append
            
    def __getitem__(self, index):        
        mask = self.mask_labels[index]
        gender = self.gender_labels[index]
        age = self.age_labels[index]
        
        image = cv2.imread(self.img_paths[index])
        label = self.get_multi_class_label(mask, gender, age)
        
        if self.transform:
            image = self.transform(image = image)
            
        return image, label
    
    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask = (multi_class_label // 6) % 3
        gender = (multi_class_label // 3) % 2
        age = multi_class_label % 3

        return mask, gender, age

    
    def split_profiles(self, profiles):
        length = len(profiles)
        n_val = int(length * self.val_ratio)
        
        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        
        return {
            "train": train_indices,
            "val": val_indices
        }
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        
        splited_profiles = self.split_profiles(profiles)
        
        count = 0
        for phase, indices in splited_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_dir = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_dir):
                    if file_name.startswith("."):
                        continue
                    _file_name, ext = os.path.splitext(file_name)
                    img_path = os.path.join(self.data_dir, profile, file_name)
                    self.img_paths.append(img_path)
                    
                    mask = self.get_mask_label(img_path)
                    gender = self.get_gender_label(img_path)
                    age = self.get_age_label(img_path)
                    
                    self.mask_labels.append(mask)
                    self.gender_labels.append(gender)
                    self.age_labels.append(age)
                    
                    self.indices[phase].append(count)
                    count += 1
                    
    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
        
    def get_mask_label(self, path):
        profile, mask = path.split('/')[-2:]
        mask = os.path.splitext(mask)[0]
        if mask[:-1].lower() == "mask":
            return 0
        elif mask.lower() == "incorrect_mask":
            return 1
        elif mask.lower() == "normal":
            return 2
        else:
            raise ValueError(f"File name({mask}) should be one of maskN, incorrect_mask, normal.")

    def get_gender_label(self, path):
        profile, mask = path.split('/')[-2:]
        id, gender, race, age = profile.split("_")
        if gender.lower() == "male":
            return 0
        elif gender.lower() == "female":
            return 1
        else:
            raise ValueError(f"Directory name({profile}) should be id_gender_race_age")
            
    def get_age_label(self, path):
        profile, mask = path.split('/')[-2:]
        id, gender, race, age = profile.split("_")
        try:
            age = int(age)
        except:
            raise ValueError(f"Age of directory name({profile}) should be numeric")
            
        if 0<age<30:
            return 0
        elif age<60:
            return 1
        else:
            return 2
        
    def get_multi_class_label(self, mask, gender, age):
        return mask * 6 + gender * 3 + age
    
    def set_transform(self, transform):
        self.transform = transform
        
class MaskDataset(MaskClassificationDataset):
    def __init__(self, data_dir='/opt/ml/input/data/train/images/', val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        mask = self.mask_labels[index]
        
        if self.transform:
            image = self.transform(image=image)
        
        return image, mask
    
class GenderDataset(MaskClassificationDataset):
    def __init__(self, data_dir='/opt/ml/input/data/train/images/', val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        gender = self.gender_labels[index]
        
        if self.transform:
            image = self.transform(image=image)
        
        return image, gender
    
class AgeDataset(MaskClassificationDataset):
    def __init__(self, data_dir='/opt/ml/input/data/train/images/', val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        age = self.age_labels[index]
        
        if self.transform:
            image = self.transform(image=image)
        
        return image, age
    
class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = A.Compose([
            A.Resize(512, 384),
            A.Normalize(mean=0.5, std=0.2),
            transforms.ToTensorV2(transpose_mask=True)
        ])
        # paths = os.path.listdir(eval_dir)
        # self.img_paths = [path for path in paths if not path.startswith(".")]
        
        
    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])

        image = self.transform(image=image)
        
        return image
    
    def __len__(self):
        return len(self.img_paths)
    
        
        
        
        
        
    