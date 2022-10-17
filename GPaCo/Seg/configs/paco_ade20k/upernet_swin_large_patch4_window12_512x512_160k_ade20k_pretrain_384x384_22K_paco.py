_base_ = [
    '../swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
    'pretrain_384x384_1K.py'
]

model = dict(
    pretrained='pretrain/swin_large_patch4_window12_384_22k.pth',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=150, type='UPerHead_paco', alpha=0.01, temperature=0.07, K=8192, freq_file="stats/ade20k_image_label_stats.txt"),
    auxiliary_head=dict(in_channels=768, num_classes=150))
data = dict(samples_per_gpu=2)

