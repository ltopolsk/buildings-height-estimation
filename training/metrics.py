import torch

def iou_score(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)
    iou = 0.0

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0:
            iou += intersection / union

    return iou / num_classes
