_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes_paco.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101), decode_head=dict(is_paco=True, alpha=0.02, temperature=0.07, K=8192))
