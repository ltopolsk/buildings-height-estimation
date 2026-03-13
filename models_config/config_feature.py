_base_ = ['./config_early.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='DualResNetFeatureFusion',
        depth=50
    )
)

load_from = 'solov2_feature_fusion_init.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend', 
        init_kwargs=dict(
            project='solov2-seg-and-height', 
            name='feature_fusion_norm'
        )
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)