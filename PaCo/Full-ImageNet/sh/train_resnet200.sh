python paco_imagenet.py \
  --arch resnet200 \ 
  --dataset imagenet \
  --data /home/sharedir/research/ImageNet/ \
  --alpha 0.05 \
  --beta 1.0 \
  --wd 1e-4 \
  --gamma 1.0 \
  --mark paco_r200_fullImageNet \
  --lr 0.04 \
  -b 256 \
  --moco-t 0.2 \
  --aug randclsstack_sim \
  --rand_m 12 \
  --rand_n 3 \
  --fp16 \
  --epochs 400
