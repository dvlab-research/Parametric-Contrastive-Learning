#!/bin/bash
#SBATCH --job-name=swintiny
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=swinbase.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint=3090
#SBATCH -p dvlab

source activate py3.8_pt1.8.1 
./tools/dist_train.sh configs/paco_ade20k/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_paco.py 8 
