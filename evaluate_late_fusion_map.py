import os
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from mmdet.apis import init_detector
from mmengine.dataset import Compose, pseudo_collate
from mmdet.evaluation.metrics import CocoMetric
from mmengine.structures import InstanceData
from torchvision.ops import nms

config_rgb = 'models_config/config_late_rgb.py'
checkpoint_rgb = 'runs/late_fusion_rgb/epoch_50.pth'

config_sar = 'models_config/config_late_sar.py'
checkpoint_sar = 'runs/late_fusion_sar/epoch_37.pth'

coco_json_path = 'dataset/annotations/val.json'
base_dir = 'dataset/val' 
device = 'cuda:0'

print("[*] Models initializing ...")
model_rgb = init_detector(config_rgb, checkpoint_rgb, device=device)
model_sar = init_detector(config_sar, checkpoint_sar, device=device)

cfg = model_rgb.cfg
raw_pipeline = cfg.test_dataloader.dataset.pipeline
inference_pipeline_cfg = [t for t in raw_pipeline if t['type'] != 'LoadAnnotations']
inference_pipeline = Compose(inference_pipeline_cfg)

metric = CocoMetric(ann_file=coco_json_path, metric=['segm'], format_only=False)
metric.dataset_meta = model_rgb.dataset_meta

coco = COCO(coco_json_path)

print(f"[*] Mask fusion for {len(coco.imgs)} images...")

for img_id, img_info in tqdm(coco.imgs.items()):
    filename = img_info['file_name']
    rgb_path = os.path.join(base_dir, 'rgb', filename)
    sar_path = os.path.join(base_dir, 'sar', filename)
    dsm_path = os.path.join(base_dir, 'dsm', filename)
    
    if not os.path.exists(rgb_path):
        continue
        
    data_info = dict(img_path=rgb_path, sar_path=sar_path, dsm_path=dsm_path, img_id=img_id)
    
    try:
        data = pseudo_collate([inference_pipeline(data_info)])
    except Exception as e:
        print(f"[!] Preprocessing error for {filename}: {e}")
        continue
    
    with torch.no_grad():
        d_rgb = model_rgb.data_preprocessor(data.copy(), False)
        res_rgb = model_rgb.predict(d_rgb['inputs'], d_rgb['data_samples'])[0]
        
        d_sar = model_sar.data_preprocessor(data.copy(), False)
        res_sar = model_sar.predict(d_sar['inputs'], d_sar['data_samples'])[0]
        
    inst_rgb = res_rgb.pred_instances
    inst_sar = res_sar.pred_instances
    
    boxes = torch.cat([inst_rgb.bboxes, inst_sar.bboxes])
    scores = torch.cat([inst_rgb.scores, inst_sar.scores])
    labels = torch.cat([inst_rgb.labels, inst_sar.labels])
    masks = torch.cat([inst_rgb.masks, inst_sar.masks])
    
    keep_indices = nms(boxes, scores, iou_threshold=0.5)
    
    new_instances = InstanceData()
    new_instances.bboxes = boxes[keep_indices]
    new_instances.scores = scores[keep_indices]
    new_instances.labels = labels[keep_indices]
    new_instances.masks = masks[keep_indices]
    
    merged_res = res_rgb.clone()
    merged_res.pred_instances = new_instances
    
    metric.process({}, [merged_res.to('cpu').to_dict()])

print("\n[*] Metrics evaluation...")
metrics = metric.evaluate(size=len(coco.imgs))
print("\n" + "="*50)
print("COCO mAP results (Late Fusion):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("="*50)