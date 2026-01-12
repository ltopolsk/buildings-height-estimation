import torch.nn as nn

class WeightedFusionWrapper(nn.Module):
    def __init__(self, model_rgb, model_sar, alpha_seg=0.0, alpha_dsm=0.3):
        """
        alpha_seg: Weight for SAR segmentation (0.0 = Ignore SAR, 0.5 = Average)
        alpha_dsm: Weight for SAR height (0.0 = Ignore SAR, 0.5 = Average)
        """
        super().__init__()
        self.model_rgb = model_rgb
        self.model_sar = model_sar
        self.alpha_seg = alpha_seg
        self.alpha_dsm = alpha_dsm
        
        # Freeze weights
        for param in self.model_rgb.parameters():
            param.requires_grad = False
        for param in self.model_sar.parameters():
            param.requires_grad = False

    def forward(self, batch):
        # 1. Get Independent Predictions
        seg_rgb, dsm_rgb = self.model_rgb(batch['rgb'])
        seg_sar, dsm_sar = self.model_sar(batch['sar'])
        
        # 2. Weighted Fusion for Segmentation
        # RGB * (1 - alpha) + SAR * alpha
        # Since SAR seg is bad, we expect alpha_seg to be 0.0
        final_seg = ((1 - self.alpha_seg) * seg_rgb) + (self.alpha_seg * seg_sar)
        
        # 3. Weighted Fusion for DSM
        # We give SAR a smaller vote (e.g., 0.3) to correct massive outliers
        final_dsm = ((1 - self.alpha_dsm) * dsm_rgb) + (self.alpha_dsm * dsm_sar)
        
        return final_seg, final_dsm
