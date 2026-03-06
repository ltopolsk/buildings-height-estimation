import torch
import matplotlib.pyplot as plt
import numpy as np

from models.unet_dual_sar import MultiTaskDualUnet
from datasets.RGBSARdataset import RGB_SAR_Dataset
import albumentations as A

# --- Configuration ---
CHECKPOINT_PATH = "best_models/best_model_unet_dual_eq.pth"
JSON_FILE = "track2/instances_val.json"
IMG_DIR = "new_data/val/rgb"
SAR_DIR = "new_data/val/sar"
DSM_DIR = "new_data/val/dsm"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_validation_transforms(height=512, width=512):
    """Simple transform just for validation (Resize/Crop only)"""
    return A.Compose([
        A.CenterCrop(height=height, width=width),
    ], additional_targets={'dsm': 'image', 'sar': 'image'}), None, None

def load_model():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = MultiTaskDualUnet()
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

def visualize_sample(model, dataset, index=0):
    items = dataset[index]
    for key, item in items.items():
        items[key] = item.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        seg_logits, dsm_pred = model(items)
        
        pred_mask = torch.sigmoid(seg_logits)
        pred_mask = (pred_mask > 0.5).float()
        
        pred_dsm = dsm_pred
        
    rgb_img = items['rgb'].squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    rgb_img = np.clip(rgb_img, 0, 1)

    sar_img = items['sar'].squeeze().cpu().numpy()
    
    mask_gt = items['mask'].squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    dsm_gt = items['dsm'].squeeze().cpu().numpy()
    pred_dsm = pred_dsm.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(sar_img, cmap='inferno')
    axes[0, 1].set_title("Input SAR (Heatmap)")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(mask_gt, cmap='gray')
    axes[0, 2].set_title("Ground Truth Mask")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(pred_mask, cmap='gray')
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(dsm_gt, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title("Ground Truth DSM")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(pred_dsm, cmap='viridis', vmin=0, vmax=1)
    axes[1, 2].set_title("Predicted DSM")
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'misc/visualization_{index}.png')

if __name__ == "__main__":
    dataset = RGB_SAR_Dataset(
        json_data_file=JSON_FILE,
        img_dir=IMG_DIR,
        dsm_dir=DSM_DIR,
        sar_dir=SAR_DIR,
        transfrom_getter=lambda h, w: get_validation_transforms(h, w)
    )
    model = load_model()
    indices = np.random.choice(len(dataset), 3, replace=False)
    for idx in indices:
        print(f"Visualizing Sample Index: {idx}")
        visualize_sample(model, dataset, index=idx)
