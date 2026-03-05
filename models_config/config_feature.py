_base_ = ['./config_early.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='DualResNetFeatureFusion',
        depth=50
    )
)

load_from = 'solov2_feature_fusion_init.pth'
