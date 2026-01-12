import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .utils import DecoderBlock


class FuseBlock(nn.Module):
    """
    Fuses features from RGB and SAR encoders.
    Concatenates them and uses 1x1 Conv to reduce channels back to 'out_channels'.
    """
    def __init__(self, c_rgb, c_sar, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_rgb + c_sar, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_rgb, f_sar):
        # Concatenate along channel dimension
        if f_rgb.shape[-2:] != f_sar.shape[-2:]:
            f_sar = nn.functional.interpolate(f_sar, size=f_rgb.shape[-2:], mode='nearest')
            
        x = torch.cat([f_rgb, f_sar], dim=1)
        return self.conv(x)


class MultiTaskDualUnet(nn.Module):
    def __init__(self, encoder_rgb='resnet34', encoder_sar='resnet18', encoder_weights='imagenet'):
        super().__init__()

        # --- BRANCH 1: RGB Encoder ---
        self.enc_rgb = smp.encoders.get_encoder(
            encoder_rgb, in_channels=3, weights=encoder_weights, depth=5
        )
        
        # --- BRANCH 2: SAR Encoder ---
        self.enc_sar = smp.encoders.get_encoder(
            encoder_sar, in_channels=1, weights=encoder_weights, depth=5
        )

        # Channel counts
        c_rgb = self.enc_rgb.out_channels # e.g. [3, 64, 64, 128, 256, 512]
        c_sar = self.enc_sar.out_channels # e.g. [1, 64, 64, 128, 256, 512]

        # --- Fusion Blocks ---
        # We fuse at every scale [f0, f1, f2, f3, f4, f5]
        self.fuse0 = FuseBlock(c_rgb[0], c_sar[0], c_rgb[0]) 
        self.fuse1 = FuseBlock(c_rgb[1], c_sar[1], c_rgb[1])
        self.fuse2 = FuseBlock(c_rgb[2], c_sar[2], c_rgb[2])
        self.fuse3 = FuseBlock(c_rgb[3], c_sar[3], c_rgb[3])
        self.fuse4 = FuseBlock(c_rgb[4], c_sar[4], c_rgb[4])
        self.fuse5 = FuseBlock(c_rgb[5], c_sar[5], c_rgb[5])

        # --- Shared Decoder ---
        # Exactly the same structure as your previous U-Net
        
        self.decoder_block1 = DecoderBlock(c_rgb[5], c_rgb[4], 256)
        self.decoder_block2 = DecoderBlock(256, c_rgb[3], 128)
        self.decoder_block3 = DecoderBlock(128, c_rgb[2], 64)
        self.decoder_block4 = DecoderBlock(64, c_rgb[1], 32)
        
        # self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        final_in_channels = 32 + c_rgb[0]
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # --- Heads ---
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        self.dsm_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, batch):

        if isinstance(batch, dict):
            img_rgb = batch['rgb']
            img_sar = batch['sar']
        else:
            img_rgb = batch[:, :3, :, :]
            img_sar = batch[:, 3:, :, :]
        
        input_shape = img_rgb.shape[-2:]

        # 1. Encode RGB and SAR
        f_rgb = self.enc_rgb(img_rgb)
        f_sar = self.enc_sar(img_sar)
        
        # 3. Fuse Features (Level by Level)
        f0 = self.fuse0(f_rgb[0], f_sar[0])
        f1 = self.fuse1(f_rgb[1], f_sar[1])
        f2 = self.fuse2(f_rgb[2], f_sar[2])
        f3 = self.fuse3(f_rgb[3], f_sar[3])
        f4 = self.fuse4(f_rgb[4], f_sar[4])
        f5 = self.fuse5(f_rgb[5], f_sar[5])

        # 4. Decode (Standard U-Net path using fused features)
        x = self.decoder_block1(f5, f4)
        x = self.decoder_block2(x, f3)
        x = self.decoder_block3(x, f2)
        x = self.decoder_block4(x, f1)
        
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.shape[-2:] != f0.shape[-2:]:
             x = nn.functional.interpolate(x, size=f0.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, f0], dim=1)

        x = self.final_conv(x)

        seg_logits = self.seg_head(x)
        dsm_pred = self.dsm_head(x)
        
        if seg_logits.shape[-2:] != input_shape:
            seg_logits = nn.functional.interpolate(seg_logits, size=input_shape, mode='bilinear', align_corners=False)
            dsm_pred = nn.functional.interpolate(dsm_pred, size=input_shape, mode='bilinear', align_corners=False)

        return seg_logits, dsm_pred
