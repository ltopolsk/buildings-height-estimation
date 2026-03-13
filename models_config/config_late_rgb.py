_base_ = ['./config_early.py']

model = dict(
    backbone=dict(
        in_channels=3,
        init_cfg=None
    ),
    data_preprocessor=dict(
        type='LateFusionDataPreprocessor',
        keep_channels=[0, 1, 2],
        custom_mean=[123.675, 116.28, 103.53],
        custom_std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32
    )
)

load_from = 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend', 
        init_kwargs=dict(
            project='solov2-seg-and-height', 
            name='decision_fusion_rgb_norm'
        )
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
