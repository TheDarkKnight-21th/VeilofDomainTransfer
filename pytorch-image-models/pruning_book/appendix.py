import os 

near_path = "./class_map/feature/near/"
in21k_path = "/home/dataset/imagenet21k_train/train"

for txt in os.listdir(near_path):

    if "cub" in txt : 
        with open(near_path+txt, 'r') as file:
            near_names = [line.strip() for line in file.readlines()]
        in21k = os.listdir(in21k_path)

        appendix = list(set(in21k)-set(near_names))
        with open(f'./class_map/feature/appendix/{txt.split(".txt")[0]}_only.txt', 'w', encoding='utf-8') as file:
                        for cls in appendix:
                            file.write(f'{cls}\n')
        