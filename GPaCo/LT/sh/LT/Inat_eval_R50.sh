#!/bin/bash
#SBATCH --job-name=gpaco_inat_r50
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=gpaco_inat_r50.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH --constraint=3090

source activate py3.8_pt1.8.1 
PORT=$[$RANDOM + 10000]

python paco_lt.py \
  --dataset inat \
  --arch resnet50 \
  --data /mnt/proj75/jqcui/Data/iNaturalist2018 \
  --alpha 0.05 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 1e-4 \
  --mark gpaco_inat_r50 \
  --lr 0.04 \
  --moco-t 0.2 \
  --aug randcls_sim \
  --randaug_m 10 \
  --randaug_n 2 \
  --dist-url "tcp://localhost:$PORT" \
  --num_classes 8142 \
  --epochs 400 \
  --resume ../../../pretrain_models/gpaco_r50_inat.pth.tar \
  --evaluate
