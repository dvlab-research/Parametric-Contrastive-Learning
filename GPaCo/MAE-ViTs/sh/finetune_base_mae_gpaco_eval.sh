#!/bin/bash
#SBATCH --job-name=mae_base_gpaco
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=mae_base_gpaco.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint=3090
#SBATCH -p dvlab

source activate py3.8_pt1.8.1 
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8888
python main_finetune_gpaco_eval.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --resume ../../../pretrain_models/gpaco_vitb_fullimagenet.pth.tar \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /mnt/proj75/jqcui/Data/ImageNet \
    --alpha 0.2 \
    --temperature 0.2 \
    --rand_n 12 \
    --rand_m 10 \
    --output_dir workdir/mae_base_gpaco \
    --corrupted_data /mnt/proj75/jqcui/Data/ImageNet-C/imagenet-c \
    --imagenet_r_data /mnt/proj75/jqcui/Data/ImageNet-R/imagenet-r \
    --imagenet_s_data /mnt/proj75/jqcui/Data/ImageNet-S/imagenet-s \
    --eval \
    --eval_std

