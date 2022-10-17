_base_ = [
    '../swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
    'pretrain_384x384_1K.py'
]
model = dict(pretrained='pretrain/swin_base_patch4_window12_384_22k.pth',
             decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150, type='UPerHead_paco', alpha=0.01, temperature=0.07, K=8192, freq_file="stats/ade20k_image_label_stats.txt"))

data = dict(samples_per_gpu=2)

