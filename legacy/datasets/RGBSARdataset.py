import json
import torch
import cv2
import numpy as np

from pycocotools import mask as coco_mask

from torch.utils.data import Dataset


class RGB_SAR_Dataset(Dataset):

    SAR_MAX_VAL = 5     # Estimated based on available training set

    def __init__(self, json_data_file, img_dir="rgb", dsm_dir="dsm", sar_dir="sar", transfrom_getter=None, height=512, width=512, max_building_height=50):
        self.metadata = json.load(open(json_data_file))
        self.img_dir = img_dir
        self.dsm_dir = dsm_dir
        self.sar_dir = sar_dir
        self.max_height = max_building_height

        if transfrom_getter:
            try:
                self.spatial_aug, self.pixel_aug, self.sar_aug = transfrom_getter(height, width)
            except ValueError:
                # no SAR transforms provided -> validation Dataset
                self.spatial_aug, self.pixel_aug = transfrom_getter(height, width)
                self.sar_aug = None
        else:
            self.spatial_aug, self.pixel_aug, self.sar_aug = None, None, None

        self.annotations = {}
        for ann in self.metadata["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)
    
    def __len__(self):
        return len(self.metadata['images'])

    def preprocess_sar(self, sar_img):
        """
        1. Fill NaNs
        2. Log Transform: Log(1 + pixel_value) to compress dynamic range
        3. MinMax Normalize to [0, 1]
        """
        sar_img = np.nan_to_num(sar_img)
        sar_img = np.log1p(sar_img) 
        sar_img = np.clip(sar_img, 0, self.SAR_MAX_VAL) / self.SAR_MAX_VAL
        return sar_img.astype(np.float32)

    def coco_segmentation_to_mask(self, segmentation, height, width):
        """
        segmentation: annotation['segmentation']
        """
        if isinstance(segmentation, list):
            # polygons
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles)
            mask = coco_mask.decode(rle)

        elif isinstance(segmentation, dict):
            # RLE
            mask = coco_mask.decode(segmentation)

        else:
            raise ValueError("Unknown segmentation format")

        return mask.astype(np.uint8)

    def __getitem__(self, index):

        img_metadata = self.metadata["images"][index]

        # RGB
        img_path = f"{self.img_dir}/{img_metadata['file_name']}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # SAR
        sar_path = f"{self.sar_dir}/{img_metadata['file_name']}"
        sar = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
        if self.sar_aug:
            sar = self.sar_aug(image=sar)['image']
        sar = self.preprocess_sar(sar)

        # DSM
        dsm_path = f"{self.dsm_dir}/{img_metadata['file_name']}"
        dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)

        dsm = np.nan_to_num(dsm, nan=.0, posinf=.0, neginf=.0)
        dsm[dsm<0] = 0

        # mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.annotations.get(img_metadata["id"], []):
            ann_mask = self.coco_segmentation_to_mask(
                ann["segmentation"], height, width
            )
            mask |= ann_mask

        # augmentation

        if self.spatial_aug and self.pixel_aug:
            augmented = self.spatial_aug(image=image, mask=mask, dsm=dsm, sar=sar)

            image = augmented['image']
            mask = augmented['mask']
            dsm = augmented['dsm']
            sar = augmented['sar']

            image = self.pixel_aug(image=image)['image']

    
        if sar.ndim == 2:
            sar = np.expand_dims(sar, axis=2)

        dsm_normalized = dsm / self.max_height
        dsm_normalized = np.clip(dsm_normalized, 0.0, 1.0)
        dsm_tensor = torch.from_numpy(dsm_normalized).float().unsqueeze(0)

        return {
            "rgb": torch.from_numpy(image).permute(2, 0, 1).float(),
            "sar": torch.from_numpy(sar).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "dsm": dsm_tensor
        }
