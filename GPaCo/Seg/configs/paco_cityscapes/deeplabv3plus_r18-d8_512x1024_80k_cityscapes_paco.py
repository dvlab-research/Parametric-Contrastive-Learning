_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes_paco.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        is_paco=True, alpha=0.01, temperature=0.07, K=8192
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
