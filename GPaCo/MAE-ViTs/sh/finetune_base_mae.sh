#!/bin/bash
#SBATCH --job-name=mae_base_baseline
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=mae_base_baseline.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint=3090
#SBATCH -p dvlab
#SBATCH -x proj77


source activate py3.8_pt1.8.1 
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port=8888 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /mnt/proj75/jqcui/Data/ImageNet \
    --output_dir mae_base_baseline
