import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.dense_heads import SOLOV2Head
from mmdet.models.data_preprocessors import DetDataPreprocessor

@MODELS.register_module()
class CustomSOLOV2Head(SOLOV2Head):
    def __init__(self, *args, height_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.height_loss_weight = height_loss_weight
        
        self.height_branch = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        self.loss_height = nn.L1Loss(reduction='mean')

    def loss(self, x, batch_data_samples, **kwargs):
        losses = super().loss(x, batch_data_samples, **kwargs)

        p2_features = x[0]
        height_pred = self.height_branch(p2_features)

        gt_heights = []
        for data_sample in batch_data_samples:
            gt_heights.append(data_sample.gt_height_map)
        
        gt_height_map = torch.stack(gt_heights, dim=0)

        height_pred_resized = F.interpolate(
            height_pred, 
            size=gt_height_map.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        loss_h = self.loss_height(height_pred_resized, gt_height_map)
        losses['loss_height'] = loss_h * self.height_loss_weight

        return losses
    
    def predict(self, x, batch_data_samples, rescale=False, **kwargs):
        results_list = super().predict(x, batch_data_samples, rescale=rescale, **kwargs)

        p2_features = x[0]
        height_preds = self.height_branch(p2_features) # [Batch, 1, H_feat, W_feat]

        for i, (pred_instances, data_sample) in enumerate(zip(results_list, batch_data_samples)):
            single_height_map = height_preds[i:i+1] 
            
            if rescale and 'ori_shape' in data_sample:
                target_shape = data_sample.ori_shape
            else:
                target_shape = data_sample.img_shape
                
            single_height_map_resized = F.interpolate(
                single_height_map, 
                size=target_shape, 
                mode='bilinear', 
                align_corners=False
            )
            
            data_sample.set_field(single_height_map_resized[0], 'pred_height_map')

        return results_list

@MODELS.register_module()
class CustomMultiModalDataPreprocessor(DetDataPreprocessor):
    def __init__(self, custom_mean, custom_std, **kwargs):
        kwargs.pop('mean', None)
        kwargs.pop('std', None)
        super().__init__(mean=None, std=None, **kwargs)
        
        self.register_buffer('custom_mean', torch.tensor(custom_mean).view(-1, 1, 1), False)
        self.register_buffer('custom_std', torch.tensor(custom_std).view(-1, 1, 1), False)

    def forward(self, data: dict, training: bool = False) -> dict | list:
        data = super().forward(data, training)
        
        inputs = data['inputs']
        inputs = (inputs - self.custom_mean) / self.custom_std
        data['inputs'] = inputs
        
        return data

@MODELS.register_module()
class DualResNetFeatureFusion(BaseModule):

    def __init__(self, depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
                 norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True,
                 style='pytorch', init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.rgb_backbone = ResNet(
            depth=depth, in_channels=3, num_stages=num_stages, 
            out_indices=out_indices, frozen_stages=frozen_stages,
            norm_cfg=norm_cfg, norm_eval=norm_eval, style=style)
            
        self.sar_backbone = ResNet(
            depth=depth, in_channels=1, num_stages=num_stages, 
            out_indices=out_indices, frozen_stages=frozen_stages,
            norm_cfg=norm_cfg, norm_eval=norm_eval, style=style)

        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(256 * 2, 256, kernel_size=1),
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.Conv2d(1024 * 2, 1024, kernel_size=1),
            nn.Conv2d(2048 * 2, 2048, kernel_size=1)
        ])

    def forward(self, x):
        rgb = x[:, :3, :, :]
        sar = x[:, 3:4, :, :]
        
        rgb_feats = self.rgb_backbone(rgb)
        sar_feats = self.sar_backbone(sar)
        
        fused_feats = []
        for i in range(len(rgb_feats)):
            concat_feat = torch.cat([rgb_feats[i], sar_feats[i]], dim=1)
            
            reduced_feat = self.fusion_convs[i](concat_feat)
            fused_feats.append(reduced_feat)
            
        return tuple(fused_feats)

@MODELS.register_module()
class LateFusionDataPreprocessor(DetDataPreprocessor):
    def __init__(self, custom_mean, custom_std, keep_channels, **kwargs):
        kwargs.pop('mean', None)
        kwargs.pop('std', None)
        super().__init__(mean=None, std=None, **kwargs)
        
        self.keep_channels = keep_channels
        self.register_buffer('custom_mean', torch.tensor(custom_mean).view(-1, 1, 1), False)
        self.register_buffer('custom_std', torch.tensor(custom_std).view(-1, 1, 1), False)

    def forward(self, data: dict, training: bool = False) -> dict | list:
        if isinstance(data['inputs'], list):
            data['inputs'] = [img[self.keep_channels, :, :] for img in data['inputs']]
        elif isinstance(data['inputs'], torch.Tensor):
            data['inputs'] = data['inputs'][:, self.keep_channels, :, :]
            
        data = super().forward(data, training)
        
        inputs = data['inputs']
        inputs = (inputs - self.custom_mean) / self.custom_std
        data['inputs'] = inputs
        
        return data