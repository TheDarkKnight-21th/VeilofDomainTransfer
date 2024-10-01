# pretrain arguments 주석을 참고하셔서 "필요하시면" arguments를 바꿔주세요.

# pretrain arguments (제가 실험한 hyperparameter입니다.)

MODEL="convnext_tiny.fb_in1k" # 항상 .fb_in1k를 붙여주셔야 합니다. scratch가 아닌 IN1k weight으로 initialization 해서 사전학습하기 때문입니다. (swin은 .ms_in1k 입니다. 모델마다 회사 이름이 다르니 참고 부탁드립니다.)
DATA_DIR="/home/dataset/imagenet21k_train/train" #../dataset/imagenet21k/train"  # 데이터 셋 경로 설정
BATCH_SIZE=256 # 총 배치사이즈는 "1024"로 GPU 개수 나누어준 수 입니다. EX ) <= 1024 / GPU 개수 (4)
GPU=4 # pretrain gpu 개수
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
LOG_INTERVAL=100
SCHED_ON_UPDATES="--sched-on-updates"
NUM_CLASSES=19167
PRETRAINED="--pretrained"  

#class map => pruning 된 class map 목록입니다. 그 목록에 있는 파일들을 전체 다 불러옵니다.
CLASS_MAP="${1:-../pruning_book/class_map/feature/near/IN21k_office-home_10_near.txt}" # class 맵을 다 이용하고 싶으시다면 오른쪽 명령어를 선택해주세요 ($(find ./pruning_book/class_map/feature/near -type f -name "*.txt")) # in dev
#pretrained model path

if [[ $CLASS_MAP == *".txt"* ]]; then
  CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_ddp_$(basename ${CLASS_MAP[$i]} .txt)/checkpoint-$((${EPOCHS}-1)).pth.tar"
else
  CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_ddp/checkpoint-$((${EPOCHS}-1)).pth.tar"
fi
# echo $CUSTOM_PRETRAINED_MODEL_PATH
# Downstream task 실행

#GPU 넘버 => 필요하시다면 바꿔주세요.
GPU_NUMBER2="${2:-7}"

if [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"office-home"* ]]; then

  CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py ../dataset/da/office-home/ \
      -d OfficeHome \
      -s Rw \
      -t Ar Cl Pr \
      -a $MODEL \
      --seed 0 \
      --log baseline/ \
      --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
      --log-wandb

elif [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"domainnet"* ]]; then

  CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py ../dataset/da/domainnet \
      -d DomainNet \
      -s r \
      -t c i p q s \
      -a $MODEL  \
      --seed 0 \
      --log baseline_domainnet/ \
      --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
      --log-wandb

elif [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"cub"* ]]; then
  
  CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py ../dataset/da/cub \
      -d CUB \
      -s Rw \
      -t Pr \
      -a $MODEL \
      --seed 0 \
      --log baseline_CUB/ \
      --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
      --log-wandb
fi
    