import torch
import numpy as np
from utils.postprocessing import split_touching_instances
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class SegHeightMetrics:
    def __init__(self, device='cpu'):
        
        self.device = device
        self.ap_metric = MeanAveragePrecision(iou_type="segm", class_metrics=False).to(device)
        self.delta1_correct_pixels = 0
        self.total_valid_pixels = 0

    def preprocess_pred_to_instances(self, binary_mask_tensor, prob_map_tensor):
        """
        Converts predictions to instances with REAL scores.
        prob_map_tensor: (B, 1, H, W) - Sigmoid probabilities
        """
        masks_np = binary_mask_tensor.detach().cpu().numpy().squeeze()
        probs_np = prob_map_tensor.detach().cpu().numpy().squeeze()
        
        # Handle batch size 1
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]
            probs_np = probs_np[np.newaxis, ...]
            
        batch_instances = []
        
        for i in range(masks_np.shape[0]):
            mask = (masks_np[i] > 0.5).astype(np.uint8)
            prob = probs_np[i]
            
            # Use connected components/splitting to find instances
            labels = split_touching_instances(mask)
            num_labels = labels.max() + 1
            
            masks_list = []
            boxes_list = []
            scores_list = [] 
            
            for label_id in range(1, num_labels):
                instance_bool = (labels == label_id)
                
                # 1. Create Mask
                instance_mask = torch.from_numpy(instance_bool).to(torch.bool)
                masks_list.append(instance_mask)
                
                # 2. Create Box
                y_idx, x_idx = torch.where(instance_mask)
                x1, y1 = x_idx.min(), y_idx.min()
                x2, y2 = x_idx.max(), y_idx.max()
                boxes_list.append(torch.tensor([x1, y1, x2, y2]))
                
                # 3. Calculate Score (Mean probability of the instance pixels)
                # This allows mAP to rank high-confidence buildings higher
                instance_score = prob[instance_bool].mean()
                scores_list.append(torch.tensor(instance_score, dtype=torch.float32))
                
            if len(masks_list) > 0:
                batch_instances.append({
                    "masks": torch.stack(masks_list).to(self.device),
                    "boxes": torch.stack(boxes_list).to(self.device),
                    "scores": torch.stack(scores_list).to(self.device),
                    "labels": torch.zeros(len(masks_list), dtype=torch.long).to(self.device)
                })
            else:
                batch_instances.append({
                    "masks": torch.empty((0, *mask.shape), dtype=torch.bool).to(self.device),
                    "boxes": torch.empty((0, 4)).to(self.device),
                    "scores": torch.empty(0).to(self.device),
                    "labels": torch.empty(0, dtype=torch.long).to(self.device)
                })
                
        return batch_instances

    def preprocess_gt_to_instances(self, binary_mask_tensor):
        """
        Ground Truth always has score 1.0, and we don't have a prob map for it.
        """
        # (This is identical to your original function, just named clearly for GT)
        masks_np = binary_mask_tensor.detach().cpu().numpy().squeeze()
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]
        
        batch_instances = []
        for i in range(masks_np.shape[0]):
            mask = (masks_np[i] > 0.5).astype(np.uint8)
            labels = split_touching_instances(mask)
            num_labels = labels.max() + 1
            
            masks_list = []
            boxes_list = []
            
            for label_id in range(1, num_labels):
                instance_bool = (labels == label_id)
                instance_mask = torch.from_numpy(instance_bool).to(torch.bool)
                masks_list.append(instance_mask)
                
                y_idx, x_idx = torch.where(instance_mask)
                boxes_list.append(torch.tensor([x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()]))

            if len(masks_list) > 0:
                batch_instances.append({
                    "masks": torch.stack(masks_list).to(self.device),
                    "boxes": torch.stack(boxes_list).to(self.device),
                    "labels": torch.zeros(len(masks_list), dtype=torch.long).to(self.device)
                })
            else:
                batch_instances.append({
                    "masks": torch.empty((0, *mask.shape), dtype=torch.bool).to(self.device),
                    "boxes": torch.empty((0, 4)).to(self.device),
                    "labels": torch.empty(0, dtype=torch.long).to(self.device)
                })
        return batch_instances

    def update(self, pred_height: torch.Tensor, target_height: torch.Tensor, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> None:
        """
        Update metrics with a new batch of data.
        
        Args:
            pred_height: (B, H, W) Predicted height map (nDSM).
            target_height: (B, H, W) Ground truth height map.
            pred_logits: Raw output from model
            target_mask: (B, H, W) Ground truth binary building mask (0-1).
        """
        # Delta 1
        pred_h = pred_height.detach().flatten()
        target_h = target_height.detach().flatten()
        valid_mask = target_h > 0
        
        p = pred_h[valid_mask]
        t = target_h[valid_mask]
        
        if p.numel() > 0:
            p = torch.clamp(p, min=1e-6)
            ratio = torch.max(t / p, p / t)
            correct = (ratio < 1.25).sum().item()
            self.delta1_correct_pixels += correct
            self.total_valid_pixels += p.numel()

        # AP50
        pred_probs = torch.sigmoid(pred_logits)
        pred_mask_binary = (pred_probs > 0.5).float()
        pred_instances = self.preprocess_pred_to_instances(pred_mask_binary, pred_probs)
        target_instances = self.preprocess_gt_to_instances(target_mask)
        
        self.ap_metric.update(pred_instances, target_instances)

    def compute(self):
        """
        Compute final scores averaged over all updates.
        Returns:
            dict containing 'AP50', 'Delta1', and 'Final_Score'
        """
        # Compute Delta 1
        delta1 = 0.0
        if self.total_valid_pixels > 0:
            delta1 = self.delta1_correct_pixels / self.total_valid_pixels
            
        # Compute AP50
        # map_50 key comes from torchmetrics
        ap_results = self.ap_metric.compute()
        ap50 = ap_results['map_50'].item()
        
        # Final Composite Score
        final_score = (ap50 + delta1) / 2.0
        
        return {
            "AP50": ap50,
            "Delta1": delta1,
            "Final_Score": final_score
        }

    def reset(self):
        self.ap_metric.reset()
        self.delta1_correct_pixels = 0
        self.total_valid_pixels = 0

