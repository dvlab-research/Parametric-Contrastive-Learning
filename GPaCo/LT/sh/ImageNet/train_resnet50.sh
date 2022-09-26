#!/bin/bash
#SBATCH --job-name=r50_fullimagenet_randclsstacksim102_alpha005
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=r50_fullimagenet_randclsstacksim102_alpha005.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -x proj77
#SBATCH --constraint=3090

source activate py3.8_pt1.8.1 

python paco_imagenet.py \
    --arch resnet50 \
    --dataset imagenet \
    --data /mnt/proj78/pgchen/data/ImageNet/ \
    --alpha 0.05 \
    --smooth 0.1 \
    --wd 1e-4 \
    --mark paco_r50_fullImageNet_randclsstacksim102_alpha005_8gpus \
    --lr 0.08 \
    -b 512 \
    --moco-t 0.2 \
    --aug randclsstack_sim \
    --rand_n 2 \
    --rand_m 10 \
    --epochs 400
