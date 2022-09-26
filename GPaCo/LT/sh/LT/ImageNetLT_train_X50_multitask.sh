#!/bin/bash
#SBATCH --job-name=gpaco_imagenetlt_x50_multitask
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=gpaco_imagenetlt_x50_multitask.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj77
#SBATCH --constraint=3090

source activate py3.8_pt1.8.1 

PORT=$[$RANDOM + 10000]
python multitask_lt.py \
  --dataset imagenet \
  --arch resnext50_32x4d \
  --data /mnt/proj75/jqcui/Data/ImageNet \
  --alpha 0.5 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark gpaco_imagenetlt_x50_multitask \
  --lr 0.06 \
  --moco-t 0.2 \
  --aug randcls_randclsstack \
  --randaug_m 10 \
  --randaug_n 3 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400
