import os 
from tqdm import tqdm
in21k_path = "/home/dataset/imagenet21k_train/train"
pruning_path =["./class_map/feature/near","./class_map/feature/far","./class_map/feature/only"]
dataset = "cub"

in21k_cnt=0
for pth in os.listdir(in21k_path):
    in21k_cnt += len(os.listdir(in21k_path+"/"+pth))

for txt_name in os.listdir(pruning_path):
    if dataset in txt_name:
        cnt = 0
        with open(pruning_path+"/"+txt_name, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        for class_name in lines:
            # print(class_name)
            cnt += len(os.listdir(in21k_path+"/"+str(class_name)))
        
        print(f"{txt_name} => 학습 사진 개수) {cnt} , 정제한 사진 개수 {in21k_cnt-cnt} ,학습 클래스 개수) {len(lines)}, 정제된 클래스 개수) {19167-len(lines)}")
