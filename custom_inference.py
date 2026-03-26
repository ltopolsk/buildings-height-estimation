import torch
import numpy as np
import cv2
import os
from pycocotools.coco import COCO
from mmengine.dataset import Compose, pseudo_collate
import matplotlib.pyplot as plt
from mmdet.apis import init_detector
from mmengine.dataset import Compose

def calculate_height_metrics(preds, targets):
    valid_mask = targets > 0
    preds_valid = preds[valid_mask]
    targets_valid = targets[valid_mask]
    
    if len(targets_valid) == 0:
        return {'delta_1': 0.0, 'delta_2': 0.0, 'delta_3': 0.0}
        
    preds_valid = np.clip(preds_valid, a_min=1e-6, a_max=None)
    ratio = np.maximum(preds_valid / targets_valid, targets_valid / preds_valid)
    
    return {
        'delta_1 (%)': round(np.mean(ratio < 1.25 ** 1) * 100, 2), 
        'delta_2 (%)': round(np.mean(ratio < 1.25 ** 2) * 100, 2), 
        'delta_3 (%)': round(np.mean(ratio < 1.25 ** 3) * 100, 2)
    }

# config
config_file = 'models_config/config_early.py'
checkpoint_file = 'runs/early_fusion/epoch_12.pth'
device = 'cuda:0'

print("[*] Loading weights...")
model = init_detector(config_file, checkpoint_file, device=device)
model.eval()

# Loading data
rgb_path = 'dataset/val/rgb/GF2_Brasilia_-15.8652_-47.9337.tif'
sar_path = 'dataset/val/sar/GF2_Brasilia_-15.8652_-47.9337.tif'
gt_height_path = 'dataset/val/dsm/GF2_Brasilia_-15.8652_-47.9337.tif'

# According to CustomSARBuildingDataset.parse_data_info
data_info = dict(
    img_path=rgb_path,
    sar_path=sar_path,
    dsm_path=gt_height_path,
    img_id=0
)

cfg = model.cfg
raw_pipeline = cfg.test_dataloader.dataset.pipeline

# Test stage doesn't have GT annotations to load 
inference_pipeline_cfg = [
    t for t in raw_pipeline if t['type'] != 'LoadAnnotations'
]

inference_pipeline = Compose(inference_pipeline_cfg)
data = inference_pipeline(data_info)

# Batch Simulation
data = pseudo_collate([data])

# Inference
print("[*] Prediction...")
with torch.no_grad():
    data = model.data_preprocessor(data, False)
    results = model.predict(data['inputs'], data['data_samples'])[0]

pred_instances = results.pred_instances
masks = pred_instances.masks.cpu().numpy() # [N, H, W]
scores = pred_instances.scores.cpu().numpy()

pred_height = results.pred_height_map.data.cpu().numpy().squeeze()

# Visualization
img_rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

gt_height = cv2.imread(gt_height_path, cv2.IMREAD_UNCHANGED)

# loead GT mask from COCO annotations 
coco_json_path = 'dataset/annotations/val.json' 
coco = COCO(coco_json_path)

filename = os.path.basename(rgb_path)

img_id = None
for img_data in coco.imgs.values():
    if img_data['file_name'] == filename:
        img_id = img_data['id']
        break

gt_mask = None
if img_id is not None:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    gt_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    
    for ann in anns:
        mask = coco.annToMask(ann)
        gt_mask = np.maximum(gt_mask, mask)
else:
    print(f"[!] Warning: no labels found for {filename}.")

# Predictions drawing
valid_inds = scores > 0.5
pred_masks_filtered = masks[valid_inds]

vis_pred_rgb = img_rgb.copy()
for m in pred_masks_filtered:
    color = np.random.randint(0, 255, 3).tolist()
    vis_pred_rgb[m] = vis_pred_rgb[m] * 0.5 + np.array(color) * 0.5 # pyright: ignore[reportOperatorIssue]

vis_gt_rgb = img_rgb.copy()
if gt_mask is not None:
    gt_binary = gt_mask > 0 
    vis_gt_rgb[gt_binary] = vis_gt_rgb[gt_binary] * 0.5 + np.array([0, 255, 0]) * 0.5 # pyright: ignore[reportOperatorIssue]

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

axs[0, 0].set_title("Ground Truth")
axs[0, 0].imshow(vis_gt_rgb.astype(np.uint8))
axs[0, 0].axis('off')

axs[0, 1].set_title("Height Ground Truth (DSM)")
vmax_val = np.max(gt_height) if gt_height is not None else 20.0 
if gt_height is not None:
    im01 = axs[0, 1].imshow(gt_height, cmap='jet', vmin=0, vmax=vmax_val)
    fig.colorbar(im01, ax=axs[0, 1])
axs[0, 1].axis('off')

axs[1, 0].set_title("Predictions (RGB + Masks)")
axs[1, 0].imshow(vis_pred_rgb.astype(np.uint8))
axs[1, 0].axis('off')

axs[1, 1].set_title("Estimated Heights (DSM)")
if pred_height is not None:
    im11 = axs[1, 1].imshow(pred_height, cmap='jet', vmin=0, vmax=vmax_val)
    fig.colorbar(im11, ax=axs[1, 1])
axs[1, 1].axis('off')

plt.tight_layout()
plt.savefig('inference_result_gt.png')
print("[+] Visualization saved to inference_result_gt.png")

if gt_height is not None and pred_height is not None:
    metrics = calculate_height_metrics(pred_height, gt_height)
    print(f"[+] Height metricts: {metrics}")