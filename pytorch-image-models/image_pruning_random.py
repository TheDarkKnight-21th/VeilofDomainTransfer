import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import heapq
from tqdm import tqdm
import math 
import random
from torch.utils.data import Dataset,DataLoader
import glob
from PIL import Image,ImageFile
import os 
import timm
from torchvision import transforms
import torch.nn as nn

random.seed(0)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def cosine_similarity(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def find_top_n_indices_and_values(lst, n=50):
    # 값과 인덱스를 튜플로 묶어서 처리

    lst_with_index = [(value, index) for index, value in enumerate(lst)]
    # heapq.nlargest를 사용하여 상위 n개의 값과 인덱스를 찾음
    top_n = heapq.nlargest(n, lst_with_index)
    # 결과 출력
    in21k_class_num = []
    cosine_book = {}
    word_book = {}
    up_keys = list(up.keys())

    for value, index in top_n:
    #    print(f"Index: {index}, Value: {value} ,IN22k : {book[list(up.keys())[index]]}")
        in21k_class_num.append(up_keys[index])
        cosine_book[up_keys[index]] = value
        word_book[up_keys[index]] = book[up_keys[index]][0]
    return in21k_class_num,cosine_book,word_book
def tar_set(sets):
    union_of_all_sets = set().union(*sets)

    # 겹치는 원소들만을 담을 집합입니다.
    overlapping_elements = set()

    for element in union_of_all_sets:
        # 원소가 속한 집합의 수를 셉니다.
        count = sum(element in s for s in sets)
        
        # 원소가 1개 이상의 집합에 속한다면, 겹치는 원소로 간주합니다.
        if count > 1:
            overlapping_elements.add(element)

    return overlapping_elements

def get_file_paths(directory,obj):
    # 파일 경로 패턴 (여기서는 모든 파일을 의미)
    pattern = directory + f'/{obj}/*'
    file_paths = glob.glob(pattern, recursive=True)

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform_color=None, transform_black=None):
        
        
        self.img_labels = []
        self.img_dir = []
        dataset_path = sorted(os.listdir(data_dir))
        for idx , dir in enumerate(dataset_path):
            #print(dir)
            self.img_dir.append(data_dir+"/"+dir)
        #print(len(self.img_labels),len(self.img_dir))
        self.transform_color = transform_color
        self.transform_black = transform_black

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
   
        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform_color or self.transform_black:
                try:
                    image = self.transform_color(image)
                except Exception as e:
                    image = self.transform_black(image)
        
            return image, img_path.split("/")[-1]
        
        except Exception as e:
            print(f'Skipped sample (index {idx}, file {img_path}). {str(e)}')

def extract_feature_timm(folder,in21k_path,down_feature,dataset_name,arch,num,percen):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # timm 모델 로드
    model = timm.create_model(arch, pretrained=True)
    model = model.to(device)
   

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

    

    total_img_sim_near = {}
    total_img_sim_far = {}
    total_img_sim_random = {}


    
    for down_class in tqdm(folder.keys()):

        for directory in folder[down_class]:
        
            directory = in21k_path+"/"+directory
            class_name = directory.split('/')[-1]
            img_name = []
            img_sim = {}

            dataset = CustomImageDataset(data_dir = directory, transform_color = preprocess_color, transform_black= preprocess_black)
            dataloader = DataLoader(dataset,batch_size = 128, shuffle=False,num_workers=4)
            

            pool = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                        nn.Flatten()
                    ).to(device=device)
            model.eval()
            for idx, (image, img_name) in enumerate((dataloader)):
            # 모델을 사용하여 이미지 특징 추출

                with torch.no_grad():
                    # features = model.forward_features(images_tensor)

                    output = model.forward_features(image.to(device))
                    features = pool(output)

                    for t,i in zip(features.cpu(),img_name):
                        img_sim[i] = t.numpy() 
            cosine_book = {}
            for i,t in img_sim.items():
                
                cosine_book[i] = cosine_similarity(t,down_feature[list(down_feature.keys())[down_class]])
            print()
            sorted_scores = sorted(cosine_book.items(), key=lambda x: x[1], reverse=True)
            # if len(sorted_scores) < 100:
            #     print(class_name)
            if class_name not in total_img_sim_near.keys():

                selected_tuples = random.sample(sorted_scores, num)
                total_img_sim_random[class_name] = sorted([t[0] for t in selected_tuples])
                total_img_sim_near[class_name]=sorted([t[0] for t in sorted_scores[:num]])
                total_img_sim_far[class_name]=sorted([t[0] for t in sorted_scores[-num:]])
            else:
                selected_tuples = random.sample(sorted_scores, num)
                total_img_sim_random[class_name].extend(sorted([t[0] for t in selected_tuples]))  
                total_img_sim_random[class_name] = sorted(list(set(total_img_sim_near[class_name])))

                total_img_sim_near[class_name].extend([t[0] for t in sorted_scores[:num]])
                total_img_sim_near[class_name] = sorted(list(set(total_img_sim_near[class_name])))

                total_img_sim_far[class_name].extend([t[0] for t in sorted_scores[-num:]])
                total_img_sim_far[class_name] = sorted(list(set(total_img_sim_far[class_name])))

    torch.save(total_img_sim_near,f"./pruning_book/efficient/office-home_{percen}_{num}_feature_near_only.pth")
    torch.save(total_img_sim_far,f"./pruning_book/efficient/office-home_{percen}_{num}_feature_far_only.pth")
    torch.save(total_img_sim_random,f"./pruning_book/efficient/office-home_{percen}_{num}_feature_random_only.pth")

    torch.cuda.empty_cache()
    return total_img_sim_near,total_img_sim_far 

if __name__ == '__main__':

    in21k_path = "/home/dataset/imagenet21k_train/train"
    for dataset_name in ['office-home']:#,'domainnet','cub'
    # up stream datatset-feature
        up_path = "./pruning_book/feature_book/in21k_feature_book_convnext_base_feature.pth"
        up = torch.load(up_path)
        
        # dowm stream, dataset-feature
        down_path = f"./pruning_book/feature_book/{dataset_name}_feature_book_convnext_base_feature.pth"
        down = torch.load(down_path)

        # IN21k class word (multi name)
        book_path = "./pruning_book/imagenet_winter_class_word.pth"
        book = torch.load(book_path)

        # UP-DOWN each class cosine similarity

        sim_book =[]
        down_values = list(down.values())
        up_values = list(up.values())

        for i in tqdm(range(len(down_values))):
            sim = [] 
            for j in range(len(up_values)):
                sim.append(cosine_similarity(down_values[i],up_values[j]))
            sim_book.append(sim)

        class_num = [1,2,20,2,4,40,4,8,80,5,10,100]
        sample_num = [200,100,10,200,100,10,200,100,10,200,100,10]
        for percen,num in zip(class_num,sample_num):
            
            print(f"Classes : {percen}, Sample : {num}")
            # sorted the cosine similarity in descending order
        
            # pruning_ratio = percen / num # percentage of data pruning per class

            down_class_num = {}
            down_word_book = {}
            down_cosine_book = {}

        

            for idx, sim_ in tqdm(enumerate(sim_book),total = len(sim_book)):
                down_class_num[idx], down_cosine_book[idx], down_word_book[idx] = find_top_n_indices_and_values(sim_,n=len(sim_))

            # data pruning by feature 
            
            pruning_class = []
            cross_set_near = []
            down_pruned_classes = {}
            du_check = []
            for class_id, cosine_scores in tqdm(down_cosine_book.items()):
                sorted_scores = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)
                
                skip = 0    
                cnt = 0
                class_cnt = 0
                while cnt < percen:
                    
                    class_len = len(os.listdir(in21k_path+"/"+sorted_scores[class_cnt][0]))
                    
              
                    if class_len<num:
                        skip += 1 
                        class_cnt += 1
                        continue

                    du = False
                    for idx in down_pruned_classes:
                        if sorted_scores[class_cnt][0] in down_pruned_classes[idx]:
                            du = True
                            break
                    if du == True:
                        class_cnt += 1
                        continue

                    if class_id not in down_pruned_classes.keys():
                        down_pruned_classes[class_id] = []
                    down_pruned_classes[class_id].append(sorted_scores[class_cnt][0])
                    du_check.append(sorted_scores[class_cnt][0])
                    # cross_set_near.append(sorted_scores[class_cnt][0])
                    class_cnt += 1
                    cnt += 1

            # Overlap class 밝히기
            # cross_set_near = tar_set(cross_set_near)
            down_count = len(down_pruned_classes)  # 내림차순 프루닝 결과의 클래스 개수
            du_check = list(set(du_check))


            print(f"skip class : {skip}")
            print(f"Overlap check !! The number of class must be {percen * len(down)} / check(ours) : {len(du_check)} ")
            if len(du_check) == len(down) * percen:
                print("There is not overlap :)")
            else:
                print("Tere is overlap :(")

            ###########################################################################################

            in21k_path = "/home/dataset/imagenet21k_train/train"
            arch = 'convnext_base.fb_in22k'
            directory_dataset = '/data/dataset/da/office-home/Real_World'
        


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


            total_img_sim_random={}
            for idx in (down_pruned_classes.keys()):

                for class_name in (down_pruned_classes[idx]):
                    
                    samples = os.listdir(in21k_path+"/"+str(class_name))

                    if class_name not in total_img_sim_random.keys():

                        selected_tuples = random.sample(samples, num)
                        total_img_sim_random[class_name] = sorted(selected_tuples)

                    else:
                        selected_tuples = random.sample(sorted_scores, num)
                        total_img_sim_random[class_name].extend(sorted(selected_tuples))  
                        total_img_sim_random[class_name] = sorted(list(set(total_img_sim_random[class_name])))

            torch.save(total_img_sim_random,f"./pruning_book/efficient/office-home_{percen}_{num}_feature_random_only.pth")

            # in21k_pb = extract_feature_timm(folder=down_pruned_classes,in21k_path = in21k_path,
            #                                 down_feature=down,arch=arch,dataset_name=dataset_name,
            #                                 num = num,percen=percen)


    
