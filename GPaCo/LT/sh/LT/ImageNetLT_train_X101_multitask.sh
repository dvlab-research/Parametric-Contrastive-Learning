#!/bin/bash
#SBATCH --job-name=gpaco_imagenetlt_x101
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=gpaco_imagenetlt_x101.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 
#SBATCH -w gpu51

PORT=$[$RANDOM + 10000]
source activate py3.8_pt1.8.1 

python multitask_lt.py \
  --dataset imagenet \
  --arch resnext101_32x4d \
  --data /research/dept6/jqcui/Data/ImageNet \
  --alpha 0.5 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark gpaco_imagenetlt_x101 \
  --lr 0.06 \
  --moco-t 0.2 \
  --aug randcls_randclsstack \
  --randaug_m 10 \
  --randaug_n 5 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 
