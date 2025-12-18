import json
import torch
import numpy as np

from pprint import pprint

from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

class BasicDataSet(Dataset):

    def __init__(self, json_data_file, img_dir="rgb", transfrom=None):
        self.metadata = json.load(open(json_data_file))
        self.img_dir = img_dir
        self.transform = transfrom

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
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(xy, outline=1, fill=1)
        
        return np.array(mask)
    
    def debug_polygons(self, ann):
        
        segmentation = ann['segmentation']
        for poly in segmentation:
            if not isinstance(poly, list):
                pprint(ann)
                print("Not a list:", type(poly))
                continue

            if len(poly) < 6:
                print(segmentation)
                print("Not enough coords:", poly)

            if len(poly) % 2 != 0:
                print(segmentation)
                print("Odd number of coords:", poly)

            for c in poly:
                if not isinstance(c, int):
                    print(segmentation)
                    print("Wrong type of coords:", c, type(c))


    def __getitem__(self, index):
        img_metadata = self.metadata["images"][index]
        img_path = f"{self.img_dir}/{img_metadata['file_name']}"

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        mask = np.zeros((height, width), dtype=np.uint8)

        image_id = img_metadata["id"]
        for ann in self.annotations.get(image_id, []):
            # self.debug_polygons(ann)
            ann_mask = self.coco_segmentation_to_mask(
                ann["segmentation"], height, width
            )
            mask |= ann_mask

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = pil_to_tensor(image).float() / 255.0
        mask = torch.tensor(mask).long()

        return image, mask