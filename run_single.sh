# pretrain arguments 주석을 참고하셔서 "필요하시면" arguments를 바꿔주세요.

# pretrain arguments (제가 실험한 hyperparameter입니다.)

MODEL="convnext_tiny.fb_in1k" # 항상 .fb_in1k를 붙여주셔야 합니다. scratch가 아닌 IN1k weight으로 initialization 해서 사전학습하기 때문입니다. (swin은 .ms_in1k 입니다. 모델마다 회사 이름이 다르니 참고 부탁드립니다.)
DATA_DIR=$1 #"./dataset/imagenet21k_train/train" #../dataset/imagenet21k/train"  # 데이터 셋 경로 설정
BATCH_SIZE=$5 # 총 배치사이즈는 "1024"로 GPU 개수 나누어준 수 입니다. EX ) <= 1024 / GPU 개수 (4)
GPU=1 # pretrain gpu 개수
GPU_NUMBER=$4
GRAD_ACCUM_STEPS=1 # 총 배치사이즈는 1024로 GPU VRAM이 부족하시다면 < 총 배치사이즈 = GPU 개수 x GPU 당 배치 x grad_accum_steps > 임으로 적절히 배치사이와 grad_accum_steps를 조절해서 총 배치를 맞춰주세요
OPTIMIZER="adamw"  # 모든 사전학습에 대해 adamw를 optimizer 진행합니다.
LEARNING_RATE=1e-4
WARMUP_EPOCHS=2
WARMUP_LR=1e-6
SCHEDULER="cosine"
EPOCHS=24
EXPERIMENT="UTL-21k"
WEIGHT_DECAY=5e-2
MIXUP=0.8
CUTMIX=1.0
LOG_INTERVAL=1
SCHED_ON_UPDATES="--sched-on-updates"
NUM_CLASSES=19167
PRETRAINED="--pretrained"  
DATASET=${2:-"default"}


cd pytorch-image-models
#class map => pruning 된 class map 목록입니다. 그 목록에 있는 파일들을 전체 다 불러옵니다.
CLASS_MAP=($(find ./pruning_book/class_map/feature -type f -name "*.txt" | sort -V)) # in dev

# far near threshold random 다 쓰고 싶다, 현재 위 명령어 경로에서 "near"를 지워주세요

# Class map (폴더안에 office home , dommainnet, cub 다 포함되어있습니다.)
# ./pytorch-image-models/pruning_book/class_map/feature
# ├─ appendix  # 타겟 데이터 셋의 클레스 별 prototype과 도메인이 가까운 순서대로 가지치기"한"(pruning) IN21k 클래스
# │  ├─ IN21k_cub_0.1_near_pruning.txt
# │  ├─ IN21k_cub_0.5_near_pruning.txt
# │  ├─ IN21k_cub_1_near_pruning.txt
# │  └─ ...
# ├─ far # 타겟 데이터 셋의 클레스 별 prototype과 가까운 순서대로 가지치기 된 (pruned) IN21k 클래스
# │  ├─ IN21k_cub_0.1_far.txt
# │  ├─ IN21k_cub_0.5_far.txt
# │  ├─ IN21k_cub_1_far.txt
# │  └─ ....
# ├─ near  # 타겟 데이터 셋의 클레스 별 prototype과 가까운 순서대로 가지치기 "된" (pruned) IN21k 클래스
# │  ├─ IN21k_cub_0.1_near.txt
# │  ├─ IN21k_cub_0.5_near.txt
# │  ├─ IN21k_cub_1_near.txt
# │  └─ ...
# ├─ random # 랜덤하게 가지치기 "된" (pruned) IN21k 클래스 (클래스 기준 : near)
# │  ├─ IN21k_cub_0.1_far_random.txt
# │  ├─ IN21k_cub_0.5_far_random.txt
# │  ├─ IN21k_cub_1_far_random.txt
# │  └─ ... 
# └─ threshold # 특정 cosine similarity threshold를 기준으로 가지치기 "된"(pruned) IN21k 클래스
#     ├─ IN21k_cub_0.65_threshold.txt
#     ├─ IN21k_cub_0.7_threshold.txt
#     ├─ IN21k_cub_0.75_threshold.txt
#     └─ ...


#GPU 넘버 => 필요하시다면 바꿔주세요.
GPU_NUMBER1=$GPU_NUMBER
length=${#CLASS_MAP[@]}


for ((i=0; i<${length}; i++)); do
  #pretrain 실행
  cd pytorch-image-models
  CUDA_VISIBLE_DEVICES=$GPU_NUMBER1 python train.py \
      --model $MODEL \
      --data-dir $DATA_DIR \
      --dataset $DATASET \
      --class-map ${CLASS_MAP[$i]} \
      --batch-size $BATCH_SIZE \
      --grad-accum-steps $GRAD_ACCUM_STEPS \
      --opt $OPTIMIZER \
      --lr $LEARNING_RATE \
      --warmup-epochs $WARMUP_EPOCHS \
      --warmup-lr $WARMUP_LR \
      --sched $SCHEDULER \
      --epochs $EPOCHS \
      --experiment $EXPERIMENT \
      --weight-decay $WEIGHT_DECAY \
      --mixup $MIXUP \
      --cutmix $CUTMIX \
      --log-interval $LOG_INTERVAL \
      $SCHED_ON_UPDATES \
      --num-classes $NUM_CLASSES \
      $PRETRAINED \
      --log-wandb

  cd ..

  cd Benchmark_Domain_Transfer
  pwd
  #pretrained model path

  if [[ ${CLASS_MAP[i]} == *".txt"* ]]; then
    CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_single_$(basename ${CLASS_MAP[$i]} .txt)/checkpoint-$((${EPOCHS}-1)).pth.tar"
  else
    CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_single/checkpoint-$((${EPOCHS}-1)).pth.tar"
  fi
  
  # Downstream task 실행

  DA_PATH=$3

  OFFICEHOME_PATH="${DA_PATH}/office-home/"
  DOMIANNET_PATH="${DA_PATH}/dommainnet/"
  CUB_PATH="${DA_PATH}/cub/"

  #GPU 넘버 => 필요하시다면 바꿔주세요.
  GPU_NUMBER2=$GPU_NUMBER
  if [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"office-home"* ]]; then

    CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $OFFICEHOME_PATH \
        -d OfficeHome \
        -s Rw \
        -t Ar Cl Pr \
        -a $MODEL \
        --seed 0 \
        --log baseline/ \
        --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
        --log-wandb

  elif [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"domainnet"* ]]; then

    CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $DOMIANNET_PATH \
        -d DomainNet \
        -s r \
        -t c i p q s \
        -a $MODEL  \
        --seed 0 \
        --log baseline_domainnet/ \
        --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
        --log-wandb

  elif [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"cub"* ]]; then
    
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $CUB_PATH \
        -d CUB \
        -s Rw \
        -t Pr \
        -a $MODEL \
        --seed 0 \
        --log baseline_CUB/ \
        --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
        --log-wandb
  fi
    
  cd ..

done
