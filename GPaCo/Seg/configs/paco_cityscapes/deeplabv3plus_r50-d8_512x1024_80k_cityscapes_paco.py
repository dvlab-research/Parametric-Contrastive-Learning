_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    decode_head=dict(type="DepthwiseSeparableASPPHead", is_paco=True, alpha=0.03, temperature=0.07, K=8192))

