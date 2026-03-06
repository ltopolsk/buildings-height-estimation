import wandb
import torch
import numpy as np
import cv2

# Standard ImageNet constants for denormalization
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize(img):
    """
    Denormalizes image: (Input * Std) + Mean.
    Expects shape [H, W, 3] or [3, H, W].
    Returns [H, W, 3] in range [0, 1].
    """
    img = img.cpu().numpy()
    
    # Handle [3, H, W] -> [H, W, 3]
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        
    img = (img * STD) + MEAN
    return np.clip(img, 0, 1)

def apply_heatmap(dsm_tensor_meters: torch.Tensor, max_height=50.0):
    """
    Converts a 1-channel DSM tensor (in meters) into an RGB Heatmap.
    """
    dsm_arr = dsm_tensor_meters.detach().cpu().numpy()
    
    dsm_arr = dsm_arr.squeeze()
    dsm_norm = np.clip(dsm_arr / max_height, 0, 1)
    
    dsm_uint8 = (dsm_norm * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(dsm_uint8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def log_predictions(rgb: torch.Tensor, sar: torch.Tensor, mask_gt: torch.Tensor, mask_pred: torch.Tensor, dsm_gt: torch.Tensor, dsm_pred: torch.Tensor, max_height=50.0):
    """
    rgb: [B, 3, H, W]
    sar: [B, 1, H, W] or None
    mask_gt: [B, H, W]
    mask_pred: [B, H, W] (Binary 0/1 Expected based on Trainer)
    dsm_gt: [B, H, W] (In Meters)
    dsm_pred: [B, H, W] (In Meters)
    """
    
    # Move to CPU once
    rgb = rgb.detach().cpu()
    mask_gt = mask_gt.detach().cpu()
    mask_pred = mask_pred.detach().cpu()
    dsm_gt = dsm_gt.detach().cpu()
    dsm_pred = dsm_pred.detach().cpu()

    if mask_gt.ndim == 4:
        mask_gt = mask_gt.squeeze(1)
    if mask_pred.ndim == 4:
        mask_pred = mask_pred.squeeze(1)

    # Handle SAR optionality for the loop
    if sar is not None:
        sar = sar.detach().cpu()
        sar_iter = sar
    else:
        sar_iter = [None] * len(rgb) # Dummy iterator

    seg_logs = []
    dsm_logs = []
    sar_logs = []

    # FIX: Safe iteration that won't crash if SAR is None
    for i, (rgb_img, sar_img) in enumerate(zip(rgb, sar_iter)):
        
        # 1. SAR Visualization
        if sar_img is not None:
            sar_numpy = sar_img.numpy().squeeze() # [H, W]
            
            # Contrast stretching for visibility
            s_min, s_max = sar_numpy.min(), sar_numpy.max()
            if s_max > s_min: 
                sar_numpy = (sar_numpy - s_min) / (s_max - s_min)
            
            sar_uint8 = (sar_numpy * 255).astype(np.uint8)
            sar_vis = cv2.applyColorMap(sar_uint8, cv2.COLORMAP_INFERNO)
            sar_vis = cv2.cvtColor(sar_vis, cv2.COLOR_BGR2RGB)
            
            sar_logs.append(wandb.Image(sar_vis, caption=f"SAR Input {i}"))

        # 2. RGB Visualization (Denormalized)
        vis_img = denormalize(rgb_img)

        # 3. Mask Visualization
        # We assume mask_pred is already binary from the trainer
        m_gt = mask_gt[i].numpy()
        m_pred = mask_pred[i].numpy()
        
        seg_logs.append(wandb.Image(
            vis_img,
            caption=f"Seg Sample {i}",
            masks={
                "ground_truth": {"mask_data": m_gt, "class_labels": {0:"Bg", 1:"Bldg"}},
                "prediction":   {"mask_data": m_pred, "class_labels": {0:"Bg", 1:"Bldg"}}
            }
        ))

        # 4. DSM Visualization
        # Pass max_height explicitly to fix the overflow bug
        d_gt_vis = apply_heatmap(dsm_gt[i], max_height=max_height)
        d_pred_vis = apply_heatmap(dsm_pred[i], max_height=max_height)
        
        dsm_combined = np.hstack((d_gt_vis, d_pred_vis))
        
        dsm_logs.append(wandb.Image(
            dsm_combined,
            caption=f"DSM (L: GT, R: Pred) {i}"
        ))

    log_dict = {
        "segmentation_examples": seg_logs,
        "dsm_examples": dsm_logs
    }
    
    if sar_logs:
        log_dict["sar_examples"] = sar_logs
        
    wandb.log(log_dict)
