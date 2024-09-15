LR_NUM=(1e-4)
WARM_NUM=(1e-6)
WARM_EPOCHS=(2)
SCHED=(cosine)
# 배열의 길이
length=${#LR_NUM[@]}

# 배열을 동시에 순회
for ((i=0; i<${length}; i++)); do
    
    CUDA_VISIBLE_DEVICES="GPU NUMBER를 지정해주세요" ./distributed_train.sh 4 \
        --model convnext_tiny.fb_in1k \
        --data-dir 데이터 셋 설정해주세요! \
        --batch-size 256 \
        --grad-accum-steps 1 \
        --opt adamw \
        --lr ${LR_NUM[i]} \
        --warmup-epochs ${WARM_EPOCHS[i]}\
        --warmup-lr ${WARM_NUM[i]} \
        --sched ${SCHED[i]} \
        --epochs 24 \
        --experiment UTL-21k \
        --weight-decay 5e-2 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --log-interval 100 \
        --sched-on-updates \
        --num-classes 19167 \
        --pretrained 
done


                ####### script 설명서 #######
# ******************************예시**************************************

# grid search를 위해서는 필요한 만큼 ',' 를 넣지않고 하이퍼파라미터 값을 추가하시면 됩니다.
# ******************* 예시 *******************
# LR_NUM=(1e-4 1e-5 1e-6 1e-7)
# WARM_NUM=(1e-6 1e-7 1e-8 1e-9)
# WARM_EPOCHS=(2 3 4 5)
# SCHED=(cosine step cosin step cosine)

# # 배열의 길이
# length=${#LR_NUM[@]}

# CUDA_VISIBLE_DEVICES="0,1,2,3" ./distributed_train.sh 4 \
#     --model convnext_tiny.fb_in1k \ # 항상 .fb_in1k를 붙여주셔야 합니다. scratch가 아닌 IN1k weight으로 initialization 해서 사전학습하기 때문입니다. (swin은 .ms_in1k 입니다. 모델마다 회사 이름이 다름니 참고 부탁드립니다.)
#     --data-dir /home/dataset/imagenet21k_train/train \
#     --batch-size 256 \ # 총 배치사이즈는 "1024"로 GPU 개수 나누어서 해주시면 됩니다. EX ) <= 1024 / GPU 개수 (4)
#     --grad-accum-steps 1 \  # 총 배치사이즈는 1024로 GPU VRAM이 부족하시다면 < 총 배치사이즈 = GPU 개수 x GPU 당 배치 x grad_accum_steps > 임으로 적절히 배치사이와 grad_accum_steps를 조절해서 총 배치를 맞춰주세요
#     --opt adamw \ # 모든 사전학습에 대해 adamw를 optimizer 진행합니다.
#     --lr ${LR_NUM[i]} \
#     --warmup-epochs ${WARM_EPOCHS[i]}\
#     --warmup-lr ${WARM_NUM[i]} \
#     --sched cosine \
#     --epochs 24 \
#     --experiment name \ 자신만의 이름을 붙여주세요
#     --weight-decay 5e-2 \
#     --mixup 0.8 \
#     --cutmix 1.0 \
#     --log-interval 100 \
#     --sched-on-updates \ default가 epoch당 lr을 업데이트 함으로 step마다 lr을 변하게 해주시려면 sched_on_updates를 선언해주셔야 합니다.
#     --num-classes 1000 \ 데이터 별 class 개수 imagenet-reiszed = 10450, imagenet21k(winter) = 19167, imagenet21k(fall) = 21841, imagenet1k = 1000
#     --pretrained \
#     --log-wandb \ wandb 사용하시면 써주세요!




