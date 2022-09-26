#!/bin/bash
#SBATCH --job-name=mae_noaug_pacov3_alpha02_t02_200epochs
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=maebase_pacov3_alpha02_t02_200e.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint=3090
#SBATCH -p dvlab


source activate py3.8_pt1.8.1 

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8888
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_pacov3_eval.py \
python main_finetune_pacov3_eval.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --resume workdir/pacov3_nomix_alpha0.2_t0.2/checkpoint-99.pth \
    --epochs 200 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --dist_eval --data_path /mnt/proj75/jqcui/Data/ImageNet \
    --alpha 0.2 \
    --temperature 0.2 \
    --output_dir workdir/pacov3_nomix_alpha0.2_t0.2 \
    --corrupted_data /mnt/proj75/jqcui/Data/ImageNet-C/imagenet-c \
    --eval \
    --eval_c 
