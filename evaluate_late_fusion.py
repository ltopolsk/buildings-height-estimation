import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from mmdet.apis import init_detector
from mmengine.dataset import Compose, pseudo_collate

config_rgb = 'models_config/config_late_rgb.py'
checkpoint_rgb = 'runs/late_fusion_rgb/epoch_12.pth'

config_sar = 'models_config/config_late_sar.py'
checkpoint_sar = 'runs/late_fusion_sar/epoch_12.pth'

coco_json_path = 'dataset/annotations/val.json'
base_dir = 'dataset/val' 
device = 'cuda:0'

print("[*] RGB model inicialization ...")
model_rgb = init_detector(config_rgb, checkpoint_rgb, device=device)

print("[*] SAR model inicialization ...")
model_sar = init_detector(config_sar, checkpoint_sar, device=device)

cfg = model_rgb.cfg
raw_pipeline = cfg.test_dataloader.dataset.pipeline
inference_pipeline_cfg = [t for t in raw_pipeline if t['type'] != 'LoadAnnotations']
inference_pipeline = Compose(inference_pipeline_cfg)

coco = COCO(coco_json_path)

total_valid_pixels = 0
total_delta1, total_delta2, total_delta3 = 0, 0, 0
total_mse = 0.0

print(f"[*] Decision Fusion evaluation on {len(coco.imgs)} images ...")

for img_id, img_info in tqdm(coco.imgs.items(), desc="Inferencja Late Fusion"):
    filename = img_info['file_name']
    
    rgb_path = os.path.join(base_dir, 'rgb', filename)
    sar_path = os.path.join(base_dir, 'sar', filename)
    dsm_path = os.path.join(base_dir, 'dsm', filename)
    
    if not os.path.exists(rgb_path) or not os.path.exists(dsm_path):
        continue
        
    data_info = dict(img_path=rgb_path, sar_path=sar_path, dsm_path=dsm_path, img_id=img_id)
    
    try:
        data_for_rgb = pseudo_collate([inference_pipeline(data_info)])
        data_for_sar = pseudo_collate([inference_pipeline(data_info)])
    except Exception as e:
        print(f"\n[!] Preprocessing error for {filename}: {e}")
        continue
    
    with torch.no_grad():
        d_rgb = model_rgb.data_preprocessor(data_for_rgb, False)
        res_rgb = model_rgb.predict(d_rgb['inputs'], d_rgb['data_samples'])[0]
        
        d_sar = model_sar.data_preprocessor(data_for_sar, False)
        res_sar = model_sar.predict(d_sar['inputs'], d_sar['data_samples'])[0]
        
    h_rgb = res_rgb.pred_height_map.data.cpu().numpy().squeeze() if hasattr(res_rgb, 'pred_height_map') else None
    h_sar = res_sar.pred_height_map.data.cpu().numpy().squeeze() if hasattr(res_sar, 'pred_height_map') else None
    gt_height = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    
    if h_rgb is not None and h_sar is not None and gt_height is not None:
        pred_height_fused = (h_rgb + h_sar) / 2.0
        
        valid_mask = gt_height > 0
        p = pred_height_fused[valid_mask]
        g = gt_height[valid_mask]
        
        if len(g) == 0:
            continue
            
        p = np.clip(p, a_min=1e-6, a_max=None)
        
        ratio = np.maximum(p / g, g / p)
        total_delta1 += np.sum(ratio < 1.25**1)
        total_delta2 += np.sum(ratio < 1.25**2)
        total_delta3 += np.sum(ratio < 1.25**3)
        total_mse += np.sum((p - g)**2)
        total_valid_pixels += len(g)

if total_valid_pixels > 0:
    print("\n" + "="*50)
    print("Decision Fusion Evaluation results")
    print("="*50)
    print(f"RMSE                    : {np.sqrt(total_mse / total_valid_pixels):.4f} m")
    print(f"Delta 1 (< 1.25)        : {(total_delta1 / total_valid_pixels) * 100:.2f}%")
    print(f"Delta 2 (< 1.25^2)      : {(total_delta2 / total_valid_pixels) * 100:.2f}%")
    print(f"Delta 3 (< 1.25^3)      : {(total_delta3 / total_valid_pixels) * 100:.2f}%")
    print("="*50)
