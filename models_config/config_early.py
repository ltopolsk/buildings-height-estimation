_base_ = ['mmdet::solov2/solov2_r50_fpn_1x_coco.py']

dataset_type = 'CustomSARBuildingDataset'

metainfo = {
    'classes': ('building',), 
    'palette': [(220, 20, 60)]
}
num_classes = 1 

custom_imports = dict(imports=['datasets.mmd_custom_dataset', 'models.custom_solov2'], allow_failed_imports=False)

model = dict(
    backbone=dict(
         in_channels=4,
        init_cfg=None
    ), 
    mask_head=dict(
        type='CustomSOLOV2Head',
        num_classes=1, 
        height_loss_weight=1.5 
    ),
    data_preprocessor=dict(
        type='CustomMultiModalDataPreprocessor',
        custom_mean=[123.675, 116.28, 103.53, 127.5],
        custom_std=[58.395, 57.12, 57.375, 58.395],
        bgr_to_rgb=False, 
        pad_size_divisor=32 
    )
)

data_root = '/home/pepsdev/mgr/buildings-height-estimation/dataset/' 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadSARAndDSMFromFile'),
    dict(type='CustomRandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='CustomRandomCrop', crop_size=(512, 512)),
    dict(type='MultiModalPixelAug'),
    dict(type='PackMultiModalInputsEarlyFusion') 
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadSARAndDSMFromFile'),
    dict(type='PackMultiModalInputsEarlyFusion') 
]

train_dataloader = dict(
    batch_size=2, 
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json', 
        data_prefix=dict(img='train/rgb/'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/rgb/'),
        test_mode=True,
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/val.json', metric=['segm'])
test_evaluator = val_evaluator

load_from = 'solov2_r50_fpn_4channel_init.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend', 
        init_kwargs=dict(
            project='solov2-seg-and-height', 
            name='early_fusion_norm'
        )
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)