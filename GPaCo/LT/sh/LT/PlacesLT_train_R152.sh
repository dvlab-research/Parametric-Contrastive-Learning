#!/bin/bash
#SBATCH --job-name=Inat
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=Inat.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 

PORT=$[$RANDOM + 10000]
#source activate py3.6pt1.7


python paco_fp16_places.py \
  --dataset places \
  --arch resnet152 \
  --data /mnt/backup2/home/sliu/jqcui/Data/Places365 \
  --alpha 0.05 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark R152_mocot0.2_sim_sim_30epochs_lr0.02_t1 \
  --lr 0.02 \
  --moco-t 0.2 \
  --aug sim_sim \
  --randaug_m 10 \
  --randaug_n 2 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 30 \
  --fp16 \
  --reload_torch pretrained_models/resnet152-394f9c45.pth
