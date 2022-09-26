#!/bin/bash
#SBATCH --job-name=x50
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=x50.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 

PORT=$[$RANDOM + 10000]
source activate py3.6pt1.7


python paco_lt.py \
  --dataset imagenet \
  --arch resnext50_32x4d \
  --data /research/dept6/jqcui/Data/ImageNet \
  --alpha 0.05 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark X50_mocot0.07_augrandclsstack_sim_400epochs_lr0.02_t1 \
  --lr 0.02 \
  --moco-t 0.07 \
  --aug randclsstack_sim \
  --randaug_m 10 \
  --randaug_n 2 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 \
  --evaluate \
  --reload pretrained_models/imagenetlt_x50.pth 
