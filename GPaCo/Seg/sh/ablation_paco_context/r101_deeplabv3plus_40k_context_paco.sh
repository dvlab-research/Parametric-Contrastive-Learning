#!/bin/bash
#SBATCH --job-name=r101
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=r101.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --constraint=3090
#SBATCH -p dvlab

source activate py3.8pt1.8.1 
./tools/dist_train.sh configs/paco_context59/deeplabv3plus_r101-d8_480x480_40k_pascal_context_59_paco.py 4 
