import torch.nn as nn

class DecisionFusionWrapper(nn.Module):
    def __init__(self, model_rgb, model_sar):
        super().__init__()
        self.model_rgb = model_rgb
        self.model_sar = model_sar
        
        # Freeze weights to save memory
        for param in self.model_rgb.parameters():
            param.requires_grad = False
        for param in self.model_sar.parameters():
            param.requires_grad = False

    def forward(self, batch):
        """
        batch: Dictionary containing 'rgb' and 'sar'
        """
        pred_seg_rgb, pred_dsm_rgb = self.model_rgb(batch['rgb'])
        
        pred_seg_sar, pred_dsm_sar = self.model_sar(batch['sar'])
        
        final_seg_logits = (pred_seg_rgb + pred_seg_sar) / 2.0
        
        final_dsm_pred = (pred_dsm_rgb + pred_dsm_sar) / 2.0
        
        return final_seg_logits, final_dsm_pred
