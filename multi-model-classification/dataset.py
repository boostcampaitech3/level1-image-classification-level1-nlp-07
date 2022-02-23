from torch.utils.data import Dataset, DataLoader
from PIL import Image

class GenderDataset(Dataset):
    def __init__(self, img_paths, transform, train):
        self.img_paths = img_paths
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.process_labels(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
            
        if self.train:
            return image,label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)
    
    def process_labels(self,path):
        info,mask = path.split('/')[-2:]
        _,gender,race,age = info.split('_')
        mask = mask.split('.')[0]
        label = {'female':0,'male':1}
        
        return label[gender]
    
class MaskDataset(Dataset):
    def __init__(self, img_paths, transform, train):
        self.img_paths = img_paths
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.process_labels(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
            
        if self.train:
            return image,label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)
    
    def process_labels(self,path):
        info,mask = path.split('/')[-2:]
        _,gender,race,age = info.split('_')
        mask = mask.split('.')[0]
        if mask[:-1]=='mask':
            mask='mask'
        label = {'incorrect_mask':0,'mask':1,'normal':2}
        
        return label[mask]
    
class AgeDataset(Dataset):
    def __init__(self, img_paths, transform, train):
        self.img_paths = img_paths
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.process_labels(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
            
        if self.train:
            return image,label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)
    
    def process_labels(self,path):
        info,mask = path.split('/')[-2:]
        _,gender,race,age = info.split('_')
        age=int(age)
        if age<30:
            return 0
        elif 30<=age<60:
            return 1
        else:
            return 2