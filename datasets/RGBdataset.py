import json
import torch
import cv2
import numpy as np

from pycocotools import mask as coco_mask

from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):


    def __init__(self, json_data_file, img_dir="rgb", dsm_dir="dsm", transfrom_getter=None, height=512, width=512, max_building_height=50):
        self.metadata = json.load(open(json_data_file))
        self.img_dir = img_dir
        self.dsm_dir = dsm_dir
        self.max_height = max_building_height

        if transfrom_getter:
            self.spatial_aug, self.pixel_aug = transfrom_getter(height, width)
        else:
            self.spatial_aug, self.pixel_aug = None, None

        self.annotations = {}
        for ann in self.metadata["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)
    
    def __len__(self):
        return len(self.metadata['images'])

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

    def polygons_to_mask(self, polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        contours = []

        for polygon in polygons:
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            contours.append(pts)
        
        cv2.fillPoly(mask, pts=contours, color=1)
        return np.array(mask)
    

    def __getitem__(self, index):
        img_metadata = self.metadata["images"][index]
        img_path = f"{self.img_dir}/{img_metadata['file_name']}"        

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        dsm_path = f"{self.dsm_dir}/{img_metadata['file_name']}"
        dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)

        dsm = np.nan_to_num(dsm, nan=.0, posinf=.0, neginf=.0)
        dsm[dsm<0] = 0

        mask = np.zeros((height, width), dtype=np.uint8)
        image_id = img_metadata["id"]
        for ann in self.annotations.get(image_id, []):
            ann_mask = self.coco_segmentation_to_mask(
                ann["segmentation"], height, width
            )
            mask |= ann_mask

        if self.spatial_aug and self.pixel_aug:
            augmented = self.spatial_aug(image=image, mask=mask, dsm=dsm)
            
            image = augmented['image']
            mask = augmented['mask']
            dsm = augmented['dsm']

            image = self.pixel_aug(image=image)['image']

        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
                
        dsm_normalized = dsm / self.max_height
        dsm_normalized = np.clip(dsm_normalized, 0.0, 1.0)
        
        dsm_tensor = torch.from_numpy(dsm_normalized).float().unsqueeze(0)

        return image_tensor, mask_tensor, dsm_tensor
