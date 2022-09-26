#!/bin/bash
#SBATCH --job-name=Inat_8gpus
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=r50_inat.log
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
  --mark R50_mocot0.2_augrandclsstack_sim_400epochs_lr0.04_8gpus \
  --lr 0.04 \
  --moco-t 0.2 \
  --aug randclsstack_sim \
  --randaug_m 10 \
  --randaug_n 2 \
  --dist-url "tcp://localhost:$PORT" \
  --num_classes 8142 \
  --epochs 400 \
  -b 256 \
  -j 56
