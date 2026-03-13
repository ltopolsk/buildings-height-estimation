import os
import cv2
import mmcv
import torch
import numpy as np
import albumentations as A
from mmdet.registry import TRANSFORMS, DATASETS
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PackDetInputs, RandomFlip
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CustomSARBuildingDataset(CocoDataset):
    METAINFO = {
        'classes': ('building',),
        'palette': [(220, 20, 60)]
    }

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)

        img_path = data_info['img_path']
        filename = os.path.basename(img_path)
        base_dir = os.path.dirname(os.path.dirname(img_path))

        data_info['sar_path'] = os.path.join(base_dir, 'sar', filename)
        data_info['dsm_path'] = os.path.join(base_dir, 'dsm', filename)

        return data_info


@TRANSFORMS.register_module()
class LoadSARAndDSMFromFile(BaseTransform):
    def transform(self, results: dict) -> dict:
        sar_path = results['sar_path']
        sar_img = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
        if sar_img is None:
            raise FileNotFoundError(f"SAR image not found: {sar_path}")

        # SAR normalization 
        clip_max = 3.5 # estimation based on training dataset
        sar_img = np.clip(sar_img, 0.0, clip_max)
        sar_img = (sar_img / clip_max) * 255.0
        sar_img = sar_img.astype(np.uint8)

        dsm_path = results['dsm_path']
        dsm_img = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
        if dsm_img is None:
            raise FileNotFoundError(f"DSM file not found: {dsm_path}")

        results['sar_img'] = sar_img
        results['gt_height_map'] = dsm_img

        return results

        return results

@TRANSFORMS.register_module()
class PackMultiModalInputsEarlyFusion(PackDetInputs):
    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        data_sample = packed_results['data_samples']
        
        if 'gt_height_map' in results:
            h_map_np = results['gt_height_map'].copy()
            h_map = torch.from_numpy(h_map_np).unsqueeze(0)
            data_sample.set_field(h_map, 'gt_height_map')
            
        if 'sar_img' in results:
            sar_img_np = results['sar_img'].copy()
            if len(sar_img_np.shape) == 2:
                sar_tensor = torch.from_numpy(sar_img_np).unsqueeze(0)
            else:
                sar_tensor = torch.from_numpy(sar_img_np).permute(2, 0, 1)
                
            data_sample.set_field(sar_tensor, 'sar_img')
            
            # --- EARLY FUSION ---
            rgb_tensor = packed_results['inputs']
            rgb_tensor = rgb_tensor[[2, 1, 0], ...]
            fused_tensor = torch.cat([rgb_tensor, sar_tensor], dim=0)
            packed_results['inputs'] = fused_tensor
            
        return packed_results

@TRANSFORMS.register_module()
class PackMultiModalInputs(PackDetInputs):

    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        data_sample = packed_results['data_samples']
        
        if 'gt_height_map' in results:
            h_map_np = results['gt_height_map'].copy()
            h_map = torch.from_numpy(h_map_np).unsqueeze(0)
            data_sample.set_field(h_map, 'gt_height_map')
            
        if 'sar_img' in results:
            sar_img_np = results['sar_img'].copy()
            if len(sar_img_np.shape) == 2:
                sar_tensor = torch.from_numpy(sar_img_np).unsqueeze(0)
            else:
                sar_tensor = torch.from_numpy(sar_img_np).permute(2, 0, 1)
                
            data_sample.set_field(sar_tensor, 'sar_img')
            
        return packed_results

@TRANSFORMS.register_module()
class MultiModalPixelAug(BaseTransform):

    def __init__(self):
        self.rgb_transform = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ], p=0.3)
        ])

        self.sar_transform = A.Compose([
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5)
        ])

    def transform(self, results: dict) -> dict:
        if 'img' in results:
            results['img'] = self.rgb_transform(image=results['img'])['image']
            
        if 'sar_img' in results:
            results['sar_img'] = self.sar_transform(image=results['sar_img'])['image']
            
        return results

@TRANSFORMS.register_module()
class CustomRandomFlip(RandomFlip):

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        
        if results.get('flip', False):
            direction = results['flip_direction']
            
            if 'sar_img' in results:
                results['sar_img'] = mmcv.imflip(results['sar_img'], direction=direction)
            if 'gt_height_map' in results:
                results['gt_height_map'] = mmcv.imflip(results['gt_height_map'], direction=direction)
                
        return results

@TRANSFORMS.register_module()
class CustomRandomCrop(BaseTransform):

    def __init__(self, crop_size):
        self.crop_size = crop_size
        
    def transform(self, results: dict) -> dict:
        h, w = results['img'].shape[:2]
        crop_h, crop_w = self.crop_size
        
        crop_h, crop_w = min(h, crop_h), min(w, crop_w)
        
        y = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        x = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        
        results['img'] = results['img'][y:y+crop_h, x:x+crop_w]
        results['img_shape'] = (crop_h, crop_w)
        
        if 'sar_img' in results:
            results['sar_img'] = results['sar_img'][y:y+crop_h, x:x+crop_w]
        if 'gt_height_map' in results:
            results['gt_height_map'] = results['gt_height_map'][y:y+crop_h, x:x+crop_w]
            results['gt_height_map_shape'] = (crop_h, crop_w)
            
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-x, -y])
            bboxes.clip_([crop_h, crop_w])
            
            valid_inds = (bboxes.widths > 0) & (bboxes.heights > 0)
            valid_inds_np = valid_inds.cpu().numpy()
            
            results['gt_bboxes'] = bboxes[valid_inds]
            
            if 'gt_bboxes_labels' in results:
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds_np]
                
            if 'gt_masks' in results:
                results['gt_masks'] = results['gt_masks'].crop(
                    np.array([x, y, x + crop_w, y + crop_h])
                )
                results['gt_masks'] = results['gt_masks'][valid_inds_np]
                
        return results
