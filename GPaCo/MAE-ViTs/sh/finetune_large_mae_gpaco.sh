#!/bin/bash
#SBATCH --job-name=mae_large_gpaco
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=mae_large_gpaco.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint=3090
#SBATCH -p dvlab
#SBATCH -x proj77

source activate py3.8pt1.8.1 
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8888 main_finetune_gpaco.py \
    --accum_iter 8 \
    --batch_size 16 \
    --model vit_large_patch16 \
    --finetune mae_pretrain_vit_large.pth \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --dist_eval --data_path /mnt/proj78/pgchen/data/ImageNet \
    --alpha 0.2 --temperature 0.2 --rand_n 12 --rand_m 10 \
    --output_dir workdir/mae_large_gpaco
