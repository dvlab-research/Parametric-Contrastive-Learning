_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/coco-stuff10k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101),
    decode_head=dict(num_classes=171, type="DepthwiseSeparableASPPHead", is_paco=True, alpha=0.01, temperature=0.07, K=8192),   auxiliary_head=dict(num_classes=171))
