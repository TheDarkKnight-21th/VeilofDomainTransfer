LP_LR_NUM=(1e-2 1e-3 1e-4)
FT_LR_NUM=(1e-4 1e-5 1e-6)
# 배열의 길이
length=${#LP_LR_NUM[@]}
# 배열을 동시에 순회
#${WARM_NUM[i]} \
for ((i=0; i<${length}; i++)); do

    CUDA_VISIBLE_DEVICES="GPU Number를 설정해주세요" ./distributed_train.sh GPU개수를 지정해주세요 \
            --model convnext_tiny.fb_in1k \
            --batch-size 256 \
            --grad-accum-steps 1 \
            --opt adamw \
            --lr ${LP_LR_NUM[i]} \
            --warmup-epochs 0\
            --epochs 3 \
            --experiment jm-pretrain-21k \
            --weight-decay 5e-2 \
            --mixup 0.8 \
            --cutmix 1.0 \
            --log-interval 100 \
            --sched-on-updates \
            --pretrained \
            --lp

    
    CUDA_VISIBLE_DEVICES="GPU Number를 설정해주세요" ./distributed_train.sh GPU개수를 지정해주세요 \
        --model convnext_tiny.fb_in1k \
        --batch-size 256 \
        --grad-accum-steps 1 \
        --opt adamw \
        --lr ${FT_LR_NUM[i]} \
        --warmup-epochs 0\
        --epochs 24 \
        --experiment jm-pretrain-21k \
        --weight-decay 5e-2 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --log-interval 100 \
        --sched-on-updates \
        --ft \
        --log-wandb
done




