import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from mmdet.apis import init_detector
from mmengine.dataset import Compose, pseudo_collate

config_file = 'models_config/config_late_rgb.py'
checkpoint_file = 'runs/late_fusion_rgb/epoch_50.pth'
coco_json_path = 'dataset/annotations/val.json'

base_dir = 'dataset/val' 
device = 'cuda:0'

print("[*] Model and pipelinie inicialization...")
model = init_detector(config_file, checkpoint_file, device=device)
cfg = model.cfg
raw_pipeline = cfg.test_dataloader.dataset.pipeline
inference_pipeline_cfg = [t for t in raw_pipeline if t['type'] != 'LoadAnnotations']
inference_pipeline = Compose(inference_pipeline_cfg)

coco = COCO(coco_json_path)

total_valid_pixels = 0
total_delta1, total_delta2, total_delta3 = 0, 0, 0
total_mse = 0.0

print(f"[*] Evaluating model {checkpoint_file} on {len(coco.imgs)} images...")

for img_id, img_info in tqdm(coco.imgs.items(), desc="Inferencja"):
    filename = img_info['file_name']
    
    rgb_path = os.path.join(base_dir, 'rgb', filename)
    sar_path = os.path.join(base_dir, 'sar', filename)
    dsm_path = os.path.join(base_dir, 'dsm', filename)
    
    if not os.path.exists(rgb_path) or not os.path.exists(dsm_path):
        continue
        
    data_info = dict(
        img_path=rgb_path,
        sar_path=sar_path,
        dsm_path=dsm_path,
        img_id=img_id
    )
    
    try:
        data = inference_pipeline(data_info)
        data = pseudo_collate([data])
    except Exception as e:
        print(f"\n[!] Preprocessing error for: {filename}: {e}")
        continue
    
    with torch.no_grad():
        data = model.data_preprocessor(data, False)
        results = model.predict(data['inputs'], data['data_samples'])[0]
        
    pred_height = results.pred_height_map.data.cpu().numpy().squeeze() if hasattr(results, 'pred_height_map') else None
    gt_height = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    
    if pred_height is not None and gt_height is not None:
        valid_mask = gt_height > 0 # pyright: ignore[reportOperatorIssue]
        p = pred_height[valid_mask]
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
    d1 = (total_delta1 / total_valid_pixels) * 100
    d2 = (total_delta2 / total_valid_pixels) * 100
    d3 = (total_delta3 / total_valid_pixels) * 100
    rmse = np.sqrt(total_mse / total_valid_pixels)
    
    print("\n" + "="*50)
    print("HEIGHT ESTIMATION EVALUATION RESULTS")
    print("="*50)
    print(f"Number of analyzed pixels: {total_valid_pixels:,}")
    print(f"RMSE                    : {rmse:.4f} m")
    print(f"Delta 1 (< 1.25)        : {d1:.2f}%")
    print(f"Delta 2 (< 1.25^2)      : {d2:.2f}%")
    print(f"Delta 3 (< 1.25^3)      : {d3:.2f}%")
    print("="*50)
else:
    print("\n[!] Error: No valid pixels found.")