# 60 이상인 사람 데이터 어그멘테이션 수행
import pandas as pd 
import os
import albumentations as A
import cv2
from PIL import Image
import numpy as np

# 60 이상 이미지 경로 리스트로 받아오기
train_dir = '/opt/ml/input/data/train'

df = pd.read_csv('/opt/ml/input/data/train/train_sort.csv') 
df=df[2510:]
image_dir = os.path.join(train_dir, 'images')

image_paths = [os.path.join(image_dir, img_id) for img_id in df.path]

img_list=[]
for i in image_paths:
    for j in os.listdir(i):
        if j[0]!='.':
            img_list.append(i+'/'+j)         
            
# https://albumentations.ai/docs/examples/example/
def augmentation_func(img_list):
    
    # augmentation 수행하여 이미지만 폴더에 저장
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.OneOf([
              A.RandomBrightnessContrast(p=1),
              A.GaussNoise(p=1),], p=0.8)])
    
    # 저장할 경로
    img_dir = '/opt/ml/code/img/'
    print(len(img_list))
    for img in img_list:
        info,mask = img.split('/')[-2:]
        pillow_image = Image.open(img)
        image = np.array(pillow_image)
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_dir+info+'-'+mask, image)
        
# augmentation_func(img_list)

# 포맷 맞춰서 train.csv에 저장/이미지 폴더 저장 (사람 id 10000부터 카운팅하기, 겹치지 않게)
arr = []
dict_arr={}
for i in os.listdir('/opt/ml/code/img'):
    if i[0]!='.':
        arr.append(i)
        # 개수 확인
        if "-".join(i.split('-')[:-1]) not in dict_arr:
            dict_arr["-".join(i.split('-')[:-1])]=1
        else:
            dict_arr["-".join(i.split('-')[:-1])]+=1
# print(dict_arr)
for i in dict_arr:
    print(i,dict_arr[i])
        
arr = sorted(arr)

print(len(arr))

df = pd.read_csv('/opt/ml/input/data/train/train.csv') 

index = 0
id={}
train_data_path = '/opt/ml/input/data/train/images/'
# train_data_path = '/opt/ml/code/sample/'
for img in arr:
    ''' data format
    003748_male_Asian_50 (사람 폴더)
    -incorrect_mask.jpg
    -mask1.jpg
    -mask2.jpg
    -mask3.jpg
    -mask4.jpg
    -mask5.jpg
    -normal.jpg
    '''
    origin_id=img.split('_')[0]
    img_name = img.split('-')[-1]
    
    if origin_id not in id.keys():
        index+=1
    # img format: 003528-1_male_Asian_60-incorrect_mask.jpg
    

    # 새로운 id 부여
    new_id = str(index+10000).zfill(6)+'_'+"_".join(img.split('_')[1:]).split('-')[0]
#     print(new_id)
    
    os.makedirs(train_data_path+new_id, exist_ok=True)
            
    pillow_image = Image.open('/opt/ml/code/img/'+img)
    image = np.array(pillow_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(train_data_path+new_id+'/'+img_name, image)
        
    if origin_id not in id.keys():
        id[origin_id]=1

        new_data = {
            'id' : [None],
            'gender' : [None],
            'race' : [None],
            'age': [None],
            'path': [new_id]
        }
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df,new_df],ignore_index=False)
  
df.to_csv("/opt/ml/input/data/train/train.csv")
# df.to_csv("/opt/ml/code/augmen.csv")


# transform = A.Compose([
#     A.CLAHE(),
#     A.RandomRotate90(),
#     A.Transpose(),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
#     A.Blur(blur_limit=3),
#     A.OpticalDistortion(),
#     A.GridDistortion(),
#     A.HueSaturationValue(),
# ])