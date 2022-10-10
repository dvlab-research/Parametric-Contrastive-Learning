#!/bin/bash
#SBATCH --job-name=cifar100-r50-multitask
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=r32.log
#SBATCH --gres=gpu:1
#SBATCH -c 4 

PORT=$[$RANDOM + 10000]
source /mnt/proj2/jqcui/ENV/py3.6pt1.81/bin/activate
python multitask_cifar.py \
  --dataset cifar100 \
  --arch resnet50 \
  --imb-factor 1.0 \
  --alpha 0.50 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark multitask_cifar100_r50 \
  --lr 0.05 \
  --moco-t 0.07 \
  --moco-k 1024 \
  --aug cifar100 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 \
  -b 128 
