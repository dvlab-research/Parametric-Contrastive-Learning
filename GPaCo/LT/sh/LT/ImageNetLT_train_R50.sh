#!/bin/bash
#SBATCH --job-name=gpaco_imagenetlt_r50
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=gpaco_imagenetlt_r50.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 

PORT=$[$RANDOM + 10000]
source activate py3.8_pt1.8.1 

python paco_lt.py \
  --dataset imagenet \
  --arch resnet50 \
  --data /mnt/proj75/jqcui/Data/ImageNet \
  --alpha 0.045 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark gpaco_imagenetlt_r50 \
  --lr 0.04 \
  --moco-t 0.2 \
  --aug randcls_randclsstack \
  --randaug_m 10 \
  --randaug_n 2 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 
