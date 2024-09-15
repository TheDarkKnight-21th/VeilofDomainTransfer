import torch
import clip
from PIL import Image
import glob
import os
import timm
from torchvision import transforms
from tqdm.auto import tqdm
import pandas as pd
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def get_file_paths(directory,obj):
    # 파일 경로 패턴 (여기서는 모든 파일을 의미)
    pattern = directory + f'/{obj}/*'
    file_paths = glob.glob(pattern, recursive=True)
    return file_paths
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform_color=None, transform_black=None):
        
        
        self.img_labels = []
        self.img_dir = []
        dataset_path = sorted(os.listdir(data_dir))
        for idx , dir in enumerate(dataset_path):
            file_list = (get_file_paths(data_dir,dir))
            #print(dir)
            self.img_dir += file_list
            self.img_labels += [str(dir)] * len(file_list)
        #print(len(self.img_labels),len(self.img_dir))
        self.transform_color = transform_color
        self.transform_black = transform_black

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
       
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.img_labels[idx]

            if self.transform_color or self.transform_black:
                try:
                    image = self.transform_color(image)
                except Exception as e:
                    image = self.transform_black(image)
        
            return image, label
        
        except Exception as e:
            print(f'Skipped sample (index {idx}, file {img_path}). {str(e)}')
        
# CLIP은 시각화 결과 imagenet22k pretrained 보다 feature extracting 결과가 직관적으로 안 좋기 때문에, 사용하지 않습니다. 
                
# def extract_feature_clip(directory):
# # CLIP 모델 로드
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)

#     # 이미지 파일 경로 리스트
#     features_list = []
    
#     dataset = CustomImageDataset(data_dir = directory, transform_color = preprocess , transform_black = None )
#     dataloader = DataLoader(dataset,batch_size = 32, shuffle=False,num_workers=0)

#     for image, label in tqdm(dataloader):
#     # 모델을 사용하여 이미지 특징 추출
#         with torch.no_grad():
#             features = model.encode_image(image)
#             features_list.append(features.cpu().numpy())
    
#     torch.cuda.empty_cache()

#     return features_list

def extract_feature_timm(directory,dataset_name,save_directory,arch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # timm 모델 로드
    model = timm.create_model(arch, pretrained=True)
    model = model.to(device)
    model.eval()

    # 이미지 전처리 : 컬러
    preprocess_color = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 이미지 전처리 : 흑백 , imagnet22k winter dataset에 흑백사진도 포함되어 있음으로 흑백 사진에 대해 따로 전처리 합니다.
    preprocess_black = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    features_list = []
    label_list = []
    features_avg_dict = {}
    prev_label = None
    current_idx = 0

    dataset = CustomImageDataset(data_dir = directory, transform_color = preprocess_color, transform_black= preprocess_black)
    dataloader = DataLoader(dataset,batch_size = 256, shuffle=False,num_workers=4)
    

    pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            ).to(device=device)
    
    for idx, (image, label) in enumerate(tqdm(dataloader)):
    # 모델을 사용하여 이미지 특징 추출

        with torch.no_grad():
            # features = model.forward_features(images_tensor)

            output = model.forward_features(image.to(device))
            features = pool(output)

            features_np = [t.numpy() for t in features.cpu()]
            features_list.extend(features_np)
            label_list.extend(list(label))

            # 현재 라벨 처리
            if prev_label is None:
                prev_label = label_list[0]

            is_last_batch = (idx == len(dataloader) - 1)
            
            #for current_idx in range(prev_idx,label_list):
            
            while(True):
                if (current_idx > (len(label_list)-1)) :
                        break
                    
                # 라벨이 변경되었거나 마지막 데이터인 경우 처리
                if label_list[current_idx] != prev_label or is_last_batch:
                    if is_last_batch:
                        print("마지막 배치입니다.",idx)
                    # 해당 라벨의 모든 feature들의 평균 계산
                    current_features_avg = np.mean(np.stack(np.array(features_list[:current_idx])),axis=0)
                    # 결과 저장
                    features_avg_dict[prev_label] = current_features_avg
                    prev_label = label_list[current_idx]
                    
                    # 처리된 feature와 라벨 pop하기 / imagenet22k은 약 1300만 장입니다. 메모리 관리를 위해 pop 합니다.  
                    for _ in range(current_idx):
                        features_list.pop(0)
                        label_list.pop(0)

                    
                    current_idx = 0
                current_idx += 1
                
        
        if idx % 2000 == 0:
            torch.save(features_avg_dict,f"{save_directory}/{dataset_name}_feature_book_{arch.split('.')[0]}_feature.pth")

    torch.cuda.empty_cache()
    print(len(features_avg_dict.keys()))
    return features_avg_dict


if __name__ == '__main__':

    directory_dataset = './dataset/da/cub/CUB_200_2011'#'./dataset/da/office-home/Real_World'# /home/dataset/imagenet21k_train/train'  # 탐색할 디렉토리 경로

    
    save_dir = './pytorch-image-models/pruning_book/feature_book'

    arch = 'convnext_base.fb_in22k'


    if "21k" in directory_dataset:
        dataset_name = "in21k"
    elif "cub" in directory_dataset:
        dataset_name = "cub"
    elif "office-home" in directory_dataset:
        dataset_name = "office-home"
    elif "domainnet" in directory_dataset:
        dataset_name = "domainnet"
    else:
        print("no matching")
    print(f"Dataset name : {dataset_name}")

    in21k_pb = extract_feature_timm(directory=directory_dataset,dataset_name=dataset_name,save_directory=save_dir,arch=arch) # data에 대한 feature 추출
    
   
    torch.save(in21k_pb, f"{save_dir}/{dataset_name}_feature_book_{arch.split('.')[0]}_feature.pth")

