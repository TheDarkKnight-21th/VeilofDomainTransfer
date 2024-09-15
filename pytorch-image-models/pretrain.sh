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

cd pytorch-image-models
#class map => pruning 된 class map 목록입니다. 그 목록에 있는 파일들을 전체 다 불러옵니다.
CLASS_MAP=($(find ./pruning_book/class_map/feature/near -type f -name "*.txt" | sort -nr)) # in dev

#GPU 넘버 => 필요하시다면 바꿔주세요.
GPU_NUMBER1="3,5,6,7"
length=${#CLASS_MAP[@]}



for ((i=0; i<${length}; i++)); do
  #pretrain 실행

  CUDA_VISIBLE_DEVICES=$GPU_NUMBER1 ./distributed_train.sh $GPU \
      --model $MODEL \
      --data-dir $DATA_DIR \
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

done
