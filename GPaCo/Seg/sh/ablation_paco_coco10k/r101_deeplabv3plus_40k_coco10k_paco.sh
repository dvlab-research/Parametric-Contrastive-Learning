#!/bin/bash
#SBATCH --job-name=r50
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=r50.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --constraint=3090
#SBATCH -p dvlab

source activate py3.8pt1.8.1 
./tools/dist_train.sh configs/paco_coco10k/deeplabv3plus_r101-d8_512x512_4x4_40k_coco-stuff10k_paco.py 4 
