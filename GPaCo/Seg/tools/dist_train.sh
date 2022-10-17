#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NUM=$((RANDOM % 40000 + 20000))
PORT=${PORT:-$NUM}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
