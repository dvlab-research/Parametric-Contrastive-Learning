#!/bin/bash
#SBATCH --job-name=x50
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=x50.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 
#SBATCH -w gpu51

PORT=$[$RANDOM + 10000]
source activate py3.6pt1.7


python paco_lt.py \
  --dataset imagenet \
  --arch resnext101_32x4d \
  --data /research/dept6/jqcui/Data/ImageNet \
  --alpha 0.05 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark X101_mocot0.2_augrandclsstack_sim_400epochs_lr0.03_rand10_3 \
  --lr 0.03 \
  --moco-t 0.2 \
  --aug randclsstack_sim \
  --randaug_m 10 \
  --randaug_n 3 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 
