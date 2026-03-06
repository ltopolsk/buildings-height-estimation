import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):

    def __init__(self, seg_weight=1.0, dsm_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
        # SmoothL1 is less sensitive to massive height outliers than MSE
        self.dsm_criterion = nn.SmoothL1Loss(reduction='none')
        
        self.seg_weight = seg_weight
        self.dsm_weight = dsm_weight

    def dice_loss(self, logits, targets, smooth=1.0):
        # Sigmoid to get probabilities [0,1]
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        
        return 1 - dice

    def forward(self, seg_logits, dsm_pred, gt_mask, gt_dsm):
        """
        seg_logits: [B, 1, H, W] from network
        dsm_pred:   [B, 1, H, W] from network
        gt_mask:    [B, H, W] Ground Truth (0 or 1)
        gt_dsm:     [B, H, W] Ground Truth heights
        """
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        
        if gt_dsm.dim() == 3:
            gt_dsm = gt_dsm.unsqueeze(1)
        
        gt_mask = gt_mask.float()
    
        bce_loss = self.bce(seg_logits, gt_mask)
        dice_loss= self.dice_loss(seg_logits, gt_mask)
        loss_seg = bce_loss + dice_loss

        pixel_loss_dsm = self.dsm_criterion(dsm_pred, gt_dsm)
        
        building_mask = (gt_mask > 0.5).float()
        
        masked_dsm_loss = pixel_loss_dsm * building_mask
        
        num_building_pixels = building_mask.sum()
        if num_building_pixels > 0:
            loss_dsm = masked_dsm_loss.sum() / num_building_pixels
        else:
            loss_dsm = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)

        total_loss = (self.seg_weight * loss_seg) + (self.dsm_weight * loss_dsm)
        
        return total_loss, loss_seg, loss_dsm
