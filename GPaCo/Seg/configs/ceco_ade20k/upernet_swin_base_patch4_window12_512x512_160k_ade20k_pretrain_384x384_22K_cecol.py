_base_ = [
    '../swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
    'pretrain_384x384_1K.py'
]
model = dict(pretrained='pretrain/swin_base_patch4_window12_384_22k.pth',
             decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150, type='UPerHead_cecol', img_cls_weight=1.0, frequency_file="stats/ade20k_image_label_stats.txt", scale=1.0,smooth=0.5))


data = dict(samples_per_gpu=4)
