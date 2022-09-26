#!/bin/bash
#SBATCH --job-name=gpaco_fullimagenet_r50
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=gpaco_fullimagenet_r50.log
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
    --data /mnt/proj75/jqcui/Data/ImageNet \
    --alpha 0.05 \
    --smooth 0.1 \
    --wd 1e-4 \
    --mark gpaco_fullimagenet_r50 \
    --lr 0.08 \
    -b 512 \
    --moco-t 0.2 \
    --aug randcls_randclsstack\
    --rand_n 2 \
    --rand_m 10 \
    --epochs 400 \
    --evaluate \
    --resume ../../../pretrain_models/gpaco_r50_fullimagenet.pth.tar 
