#!/bin/bash
#SBATCH --job-name=cifar100-r32
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=r32.log
#SBATCH --gres=gpu:1
#SBATCH -c 2 
#SBATCH -A leojia
#SBATCH -p leojia
#SBATCH -x proj77 


PORT=$[$RANDOM + 10000]

python paco_cifar.py \
  --dataset cifar100 \
  --arch resnet32 \
  --imb-factor 0.02 \
  --alpha 0.01 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark CIFAR100_imb002_R32_001_005 \
  --lr 0.05 \
  --moco-t 0.05 \
  --moco-k 1024 \
  --moco-dim 32 \
  --feat_dim 64 \
  --aug cifar100 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400
