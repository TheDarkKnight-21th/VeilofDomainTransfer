
import torch
import os 
from tqdm.auto import tqdm
from nltk.corpus import wordnet as wn
import argparse

# wordnet 설치를 위해 최초 1회 설치하시면 됩니다.
# import nltk
# nltk.download('wordnet')

# argument 수정 중
parser = argparse.ArgumentParser(description='Data Pruning')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecatedf*, use --data-dir)')
parser.add_argument('--down-dir', metavar='DIR',default='/home/dataset/da/domainnet/real',
                    help='path to downstream dataset (root dir)')
parser.add_argument('--up-dir', metavar='DIR',default='/home/dataset/imagenet21k_train/train/',
                    help='path to upstream dataset (root dir)')
parser.add_argument('--threshold', default=0.3, type=int,
                        help='threshold to path similarity.')
args = parser.parse_args()

def find_top_category(word):
    synsets = wn.synsets(word)
    if len(synsets) > 0:
        synset = synsets[0]  # 첫번째 의미를 선택
        while len(synset.hypernyms()) > 0:  # 상위어가 존재하는 동안 반복
            synset = synset.hypernyms()[0]  # 상위어로 이동
            print(synset,end="\n")
        return synset.lemmas()[0].name()  # 최상위 카테고리 반환
    else:
        return None  # WordNet에 해당 단어가 없는 경우
def is_hypernym_of(hypernym_word, hyponym_synset):
    # 상위어를 찾기
    hypernyms = hyponym_synset.hypernyms()
    if len(hypernyms) > 0:
        current_hypernym = hypernyms[0]
        # 상위어가 찾는 단어와 같은지 확인
        if hypernym_word in current_hypernym.lemma_names():
            return True
        # 상위어가 찾는 단어가 아니면 더 상위로 이동
        else:
            return is_hypernym_of(hypernym_word, current_hypernym)
    else:
        return False
    
def is_similar_word(word1, word2,threshold):
        # 두 단어의 첫 번째 synset 가져오기
        synset1 = wn.synsets(word1)
        synset2 = wn.synsets(word2)
        sim = []
        
        if len(synset1) > 0 and len(synset2) > 0:

            for syn1 in synset1:
                for syn2 in synset2:
                    sim.append(syn1.path_similarity(syn2))
        
        
        # 두 synset의 유사도 계산
            similarity = max(sim)
                #print(similarity)
            
            # 유사도가 일정 임계값(예: 0.5) 이상이면 비슷한 단어로 간주
            if similarity and similarity >= threshold:
                return True , similarity
            return False , similarity
        else:
            print(synset1,word1,synset2,word2)
            return False,-1
        
def find_word_in_list(word, list_2d):
    for i, sublist in enumerate(list_2d):
        if word in sublist:
            return i
    return -1  # 단어를 찾지 못했을 때 반환할 값

if __name__ == "__main__":
    #description_path = './pruning_book/imagenet21k_miil_tree.pth'
    in21k_wordnet_path = './pruning_book/in21k_wordnet_class.txt'
    in21k_wordnet_lemmas_path = './pruning_book/in21k_wordnet_lemmas.txt'
    
    #downstream dataset path 
    downstream_task_path = args.down_dir #"/home/dataset/da/domainnet/real" #"/home/dataset/da/office-home/Art"
    #imagenet21k winter or fall path
    in21k_path = args.up_dir

    if "office-home" in downstream_task_path or "OFFICE-HOME" in downstream_task_path : 
        down_name = "office-home"
    elif "domainnet" in downstream_task_path or "DOMAINNET" in downstream_task_path :
        down_name = "domainnet"
    elif "cub" in downstream_task_path or "CUB" in downstream_task_path : 
        down_name = "cub"

    threshold = args.threshold

    print("Down stream task : ",downstream_task_path)
    # data = torch.load(description_path)
    # in21k = list(data['class_description'].values())

    in21k = []
    in21k_word = []
    in21k_class = []
    in21k_class2 = []
    p = []

    with open(in21k_wordnet_path, 'r') as file1:
        for line in file1:
            # 줄을 ','로 분리하여 단어 추출
            line_words = line.strip()
            in21k_class.append(line_words)
    winter = os.listdir(in21k_path)

    # txt 파일 열기
    with open(in21k_wordnet_lemmas_path, 'r') as file2:
        # 각 줄을 읽기
        for idx,line in enumerate(file2):
                
            if in21k_class[idx] in winter: 
            # 줄을 ','로 분리하여 단어 추출
                in21k_class2.append(in21k_class[idx])
                line_words = line.strip().split(',')
                # 공백 제거 후 리스트에 추가
                in21k.append([word.strip() for word in line_words])
                in21k_word.extend(word.strip() for word in line_words)

    label = os.listdir(downstream_task_path)
    if "cub" in downstream_task_path or "CUB" in downstream_task_path : 
        label = ['bird'] # if dataset is CUB 

    for name in tqdm(range(len(label))):

        if label[name] == 'Desk_Lamp' :
            label[name] = 'lamp'
        for k in (in21k_word):
            sig,sim = is_similar_word(label[name], k,threshold=threshold)
            if sim == -1 :
                break
            if sig : 
                p.append(k)
    
    p = list(set(p))

    # 찾고자 하는 value
    k = []
    for name in tqdm(p):

        k.append(find_word_in_list(name, in21k))

    k = list(set(k)) # remove duplicates
    total = 0
    for num in tqdm(os.listdir(in21k_path)) :
        if ".txt" in num:
            continue
        total += len(os.listdir(in21k_path+num))
    print("Pruned class",len(k))
    pru = 0
    source_class = []
    for num in tqdm(k):

        pru += len(os.listdir(in21k_path+in21k_class2[num]))
        source_class.append(in21k_class2[num])
        
    print(f"Total : {total} , Da Duple : {pru}, Pruned : {total-pru}  ")

    target_class = os.listdir(in21k_path)

    out = set(target_class) - set(source_class)

    with open(f'./pruning_book/class_map/word_net/IN21k_{down_name}_{threshold}.txt', 'w', encoding='utf-8') as file:
        for cls in out:
            file.write(f'{cls}\n')