_base_ = ['./config_early.py']

model = dict(
    backbone=dict(
        in_channels=1,
        init_cfg=None
    ),
    data_preprocessor=dict(
        type='LateFusionDataPreprocessor',
        keep_channels=[3],
        custom_mean=[127.5],
        custom_std=[58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32
    )
)

load_from = 'solov2_sar_only_init.pth'