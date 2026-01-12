import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .utils import DecoderBlock


class MultiTaskNetSar(nn.Module):

    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=4):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=in_channels, 
            weights=encoder_weights, 
            depth=5
        )
        
        encoder_channels = self.encoder.out_channels

        # Decoder
        self.decoder_block1 = DecoderBlock(encoder_channels[5], encoder_channels[4], 256)
        self.decoder_block2 = DecoderBlock(256, encoder_channels[3], 128)
        self.decoder_block3 = DecoderBlock(128, encoder_channels[2], 64)
        self.decoder_block4 = DecoderBlock(64, encoder_channels[1], 32)
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        final_in_channels = 32 + encoder_channels[0]

        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Segmentation Head
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Height Head
        self.dsm_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, batch):

        if isinstance(batch, dict):
            x = torch.cat([batch['rgb'], batch['sar']], dim=1)
        else:
            x = batch

        # x.shape == [batch, 4, H, W]
        input_shape = x.shape[-2:]

        # Encoder
        features = self.encoder(x)

        # Decoder
        d1 = self.decoder_block1(features[5], features[4])
        d2 = self.decoder_block2(d1, features[3])
        d3 = self.decoder_block3(d2, features[2])
        d4 = self.decoder_block4(d3, features[1])
        
        x = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Check shapes for f0 concatenation (handling potential padding issues)
        if x.shape != features[0].shape:
            x = nn.functional.interpolate(x, size=features[0].shape[-2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, features[0]], dim=1)
        
        x = self.final_conv(x)

        # Heads
        seg_logits = self.seg_head(x)
        dsm_pred = self.dsm_head(x)
        
        # Final safety resize to match original input (if rounding errors occurred)
        if seg_logits.shape[-2:] != input_shape:
            seg_logits = nn.functional.interpolate(seg_logits, size=input_shape, mode='bilinear', align_corners=False)
            dsm_pred = nn.functional.interpolate(dsm_pred, size=input_shape, mode='bilinear', align_corners=False)

        return seg_logits, dsm_pred
