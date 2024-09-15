import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import heapq
from tqdm import tqdm
import math 
import random

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




if __name__ == '__main__':

    #mode = "far" # near far random
    for mode in ["feature"]:
        print("Mode : ",mode)
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

            if mode == 'threshold' :

                for thres in [0.001,0.01,0.1]:
                    down_class_num = {}
                    down_word_book = {}
                    down_cosine_book = {}

                    for idx, sim_ in tqdm(enumerate(sim_book),total = len(sim_book)):
                        down_class_num[idx], down_cosine_book[idx], down_word_book[idx] = find_top_n_indices_and_values(sim_,n=len(sim_))

                    # data pruning by feature 
                    
                    pruning_class = [] 
                    thres_dict = {outer_k: {inner_k: inner_v for inner_k, inner_v in outer_v.items() if inner_v >= thres}\
                                for outer_k, outer_v in down_cosine_book.items()}
                    for class_num in tqdm(range(len(thres_dict))):
                        pruning_class.extend(list(thres_dict[class_num].keys()))
                    pruning_class = list(set(pruning_class))
                    pruned_class = list(set(down_cosine_book[0].keys())-set(pruning_class))
                    
                    print(f"({dataset_name}, 클래스 별 pruning한 클래스 개수 : {thres}) // Total : {len(down_cosine_book[0].keys())} ,",
                        f" Pruned class (남은 클래스) : {len(pruned_class)}, Pruning class (제거한 클래스) {len(pruning_class)}")

                    # print(sorted(pruned_class))
                    pruned_class = sorted(pruned_class)
                    with open(f'./pruning_book/class_map/feature/{mode}/IN21k_{dataset_name}_{thres}_{mode}.txt', 'w', encoding='utf-8') as file:
                        for cls in pruned_class:
                            file.write(f'{cls}\n')
            else:
                for percen in [1,10,100,1000]:
                    # sorted the cosine similarity in descending order
                
                    # pruning_ratio = percen / 100 # percentage of data pruning per class


                    down_class_num = {}
                    down_word_book = {}
                    down_cosine_book = {}

                    for idx, sim_ in tqdm(enumerate(sim_book),total = len(sim_book)):
                        down_class_num[idx], down_cosine_book[idx], down_word_book[idx] = find_top_n_indices_and_values(sim_,n=len(sim_))

                    # data pruning by feature 
                    
                    pruning_class = []
                    cross_set_near = []
                    down_pruned_classes = set()
                    for class_id, cosine_scores in down_cosine_book.items():
                        sorted_scores = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)[:percen]
                        down_pruned_classes.update([score[0] for score in sorted_scores])
                        cross_set_near.append([score[0] for score in sorted_scores])

                    # 오름차순 프루닝
                    # down_pruned_classes와 up_pruned_classes의 개수를 맞추기 위한 접근
                    cross_set_near = tar_set(cross_set_near)
                    down_count = len(down_pruned_classes)  # 내림차순 프루닝 결과의 클래스 개수

                    # 오름차순 프루닝을 위한 초기 설정

                    # 선택된 클래스의 개수가 down_pruned_classes의 개수와 동일하거나 더 많은 상태
                    # 필요에 따라 up_pruned_classes에서 추가 조정을 수행할 수 있음

                    up_pruned_classes = set()
                    cross_set_far = []
                    for class_id, cosine_scores in down_cosine_book.items():
                        sorted_scores = sorted(cosine_scores.items(), key=lambda x: x[1])[:percen]  # 유사도가 낮은 하위 10개 선택
                        up_pruned_classes.update([score[0] for score in sorted_scores])
                        cross_set_far.append([score[0] for score in sorted_scores])
                        
                    cross_set_far = tar_set(cross_set_far)

                    # 내림차순 프루닝과 오름차순 프루닝 결과 개수 확인
                                      # pruning_class = list(set(pruning_class))
                    
                    pruned_class_near = list(set(down_cosine_book[0].keys())-down_pruned_classes)
                    pruned_class_far = list(set(down_cosine_book[0].keys())-up_pruned_classes)
                    pruned_class_near = sorted(pruned_class_near)
                    pruned_class_far = sorted(pruned_class_far)
                    
                    print(f"({dataset_name}, Near || 클래스 별 pruning한 클래스 개수 : {percen}) // Total : {len(down_cosine_book[0].keys())} ,",
                        f" Pruned class (남은 클래스) : {len(pruned_class_near)}, Pruning class (제거한 클래스) {len(down_pruned_classes)}",
                        f" // overlap 된 클래스 {len(cross_set_near)}")
                    print(f"({dataset_name}, FAR || 클래스 별 pruning한 클래스 개수 : {percen}) // Total : {len(down_cosine_book[0].keys())} ,",
                        f" Pruned class (남은 클래스) : {len(pruned_class_far)}, Pruning class (제거한 클래스) {len(up_pruned_classes)}",
                        f" // overlap 된 클래스 {len(cross_set_far)}")

                    selection_per_class = percen  # 초기 선택 개수 설정 (예: 10개)
                    # 초기 선택 후, selected_classes의 개수가 near_pruning_count에 도달할 때까지 각 클래스별로 1개씩 추가
                    while len(up_pruned_classes) < len(down_pruned_classes):
                        for class_id, cosine_scores in down_cosine_book.items():
                            # 이미 선택된 클래스는 넘어감
                            sorted_scores = [score for score in sorted(cosine_scores.items(), key=lambda x: x[1]) if score[0] not in up_pruned_classes]
                            
                            if sorted_scores:
                                # 추가할 클래스가 있으면, selected_classes에 추가
                                up_pruned_classes.add(sorted_scores[0][0])
                                
                                # near_pruning_count에 도달하면 반복 중단
                                if len(up_pruned_classes) >= len(down_pruned_classes):
                                    break
                    print(f"<<<<<<<<<<<<<<<<<<<조정된 오름차순(far) 프루닝 클래스 개수 : {len(up_pruned_classes)}")
                    pruned_class_far = list(set(down_cosine_book[0].keys())-up_pruned_classes)
               
                    # make random pruning class
                    in21k_class = list(book.keys())
                    random.shuffle(in21k_class)
                    random_pruned_class = in21k_class[:len(pruned_class_near)]
                    random_pruned_class_list = sorted(random_pruned_class)
                    print("랜덤 pruning class 개수 : ",len(random_pruned_class))

                    # write pruned class map text file
                    with open(f'./pruning_book/class_map/feature/near/IN21k_{dataset_name}_{percen}_near.txt', 'w', encoding='utf-8') as file:
                        for cls in pruned_class_near:
                            file.write(f'{cls}\n')

                    with open(f'./pruning_book/class_map/feature/far/IN21k_{dataset_name}_{percen}_far.txt', 'w', encoding='utf-8') as file:
                        for cls in pruned_class_far:
                            file.write(f'{cls}\n')
                    
                    # write random pruned class map text file
                    with open(f'./pruning_book/class_map/feature/random/IN21k_{dataset_name}_{percen}_near_random.txt', 'w', encoding='utf-8') as file:
                        for cls in random_pruned_class:
                            file.write(f'{cls}\n')


    
