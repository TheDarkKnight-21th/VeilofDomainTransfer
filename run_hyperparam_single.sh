# !!!!!! None Pruning !!!!!! class map이 없습니다.

# pretrain arguments 주석을 참고하셔서 "필요하시면" arguments를 바꿔주세요.

# pretrain arguments (제가 실험한 hyperparameter입니다.)

MODEL="convnext_tiny.fb_in1k" # 항상 .fb_in1k를 붙여주셔야 합니다. scratch가 아닌 IN1k weight으로 initialization 해서 사전학습하기 때문입니다. (swin은 .ms_in1k 입니다. 모델마다 회사 이름이 다르니 참고 부탁드립니다.)
DATA_DIR=$1 #"../dataset/imagenet21k_train/train" #../dataset/imagenet21k/train"  # 데이터 셋 경로 설정
BATCH_SIZE=$7 # 총 배치사이즈는 "1024"로 GPU 개수 나누어준 수 입니다. EX ) <= 1024 / GPU 개수 (4)
GPU=1 # pretrain gpu 개수
GPU_NUMBER=$6 # pretrain gpu 개수
GRAD_ACCUM_STEPS=1 # 총 배치사이즈는 1024로 GPU VRAM이 부족하시다면 < 총 배치사이즈 = GPU 개수 x GPU 당 배치 x grad_accum_steps > 임으로 적절히 배치사이와 grad_accum_steps를 조절해서 총 배치를 맞춰주세요
OPTIMIZER="adamw"  # 모든 사전학습에 대해 adamw를 optimizer 진행합니다.
SCHEDULER="cosine"
EPOCHS=24
WARMUP_EPOCHS=2
EXPERIMENT="UTL-21k"
WEIGHT_DECAY=5e-2
MIXUP=0.8
CUTMIX=1.0
LOG_INTERVAL=100
SCHED_ON_UPDATES="--sched-on-updates"
NUM_CLASSES=19167
PRETRAINED="--pretrained"  
DATASET=${2:-"default"}

# grid search parameters
LEARNING_RATE=$4 # grid search를 위해 파라미터를 추가해주세요! | ex : (1e-4 1e-6 1e-7 ...)
WARMUP_LR=$5 # grid search를 위해 파라미터를 추가해주세요! | ex : (1e-6 1e-7 1e-8 ...)

#GPU 넘버 => 필요하시다면 바꿔주세요.
GPU_NUMBER1=$GPU_NUMBER
#pretrain 실행
cd pytorch-image-models
CUDA_VISIBLE_DEVICES=$GPU_NUMBER1 python train.py\
    --model $MODEL \
    --data-dir $DATA_DIR \
    --dataset $DATASET \
    --batch-size $BATCH_SIZE \
    --grad-accum-steps $GRAD_ACCUM_STEPS \
    --opt $OPTIMIZER \
    --lr $LEARNING_RATE \
    --warmup-epochs $WARMUP_EPOCHS \
    --warmup-lr $WARMUP_LR \
    --workers 12 \
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
CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_single/checkpoint-$((${EPOCHS}-1)).pth.tar"


# Downstream task 실행

#GPU 넘버 => 필요하시다면 바꿔주세요.
GPU_NUMBER2=$GPU_NUMBER

DA_PATH=$3

OFFICEHOME_PATH="${DA_PATH}/office-home/"
DOMIANNET_PATH="${DA_PATH}/domainnet/"
CUB_PATH="${DA_PATH}/cub/"

CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $OFFICEHOME_PATH \
    -d OfficeHome \
    -s Rw \
    -t Ar Cl Pr \
    -a $MODEL \
    --seed 0 \
    --log baseline/ \
    --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
    --log-wandb



CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $DOMIANNET_PATH \
    -d DomainNet \
    -s r \
    -t c i p q s \
    -a $MODEL  \
    --seed 0 \
    --log baseline_domainnet/ \
    --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
    --log-wandb



CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $CUB_PATH \
    -d CUB \
    -s Rw \
    -t Pr \
    -a $MODEL \
    --seed 0 \
    --log baseline_CUB/ \
    --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
    --log-wandb

