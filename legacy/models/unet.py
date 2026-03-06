import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .utils import DecoderBlock


class MultiTaskUnet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=in_channels, 
            weights=encoder_weights, 
            depth=5
        )
        
        encoder_channels = self.encoder.out_channels

        self.decoder_block1 = DecoderBlock(encoder_channels[5], encoder_channels[4], 256)
        self.decoder_block2 = DecoderBlock(256, encoder_channels[3], 128)
        self.decoder_block3 = DecoderBlock(128, encoder_channels[2], 64)
        self.decoder_block4 = DecoderBlock(64, encoder_channels[1], 32)
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        final_in_channels = 32 + encoder_channels[0] # == 35 channels
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Height head
        self.dsm_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        if isinstance(x, dict):
            x = x['rgb']

        # x.shape == [batch, 3, H, W]
        input_shape = x.shape[-2:]

        features = self.encoder(x)
        x_dec = self.decoder_block1(features[5], features[4])
        x_dec = self.decoder_block2(x_dec, features[3])
        x_dec = self.decoder_block3(x_dec, features[2])
        x_dec = self.decoder_block4(x_dec, features[1])
        x_dec = self.final_upsample(x_dec)
        
        x = self.final_upsample(x)
        x = torch.cat([x_dec, features[0]], dim=1)

        x = self.final_conv(x)

        seg_logits = self.seg_head(x)
        dsm_pred = self.dsm_head(x)
        
        # Interpolate if needed
        if seg_logits.shape[-2:] != input_shape:
             seg_logits = nn.functional.interpolate(seg_logits, size=input_shape, mode='bilinear', align_corners=False)
             dsm_pred = nn.functional.interpolate(dsm_pred, size=input_shape, mode='bilinear', align_corners=False)

        return seg_logits, dsm_pred

