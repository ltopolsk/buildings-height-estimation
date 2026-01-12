import albumentations as A
import cv2

def get_transforms(height=256, width=256):
    spatial_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=height, width=width, always_apply=True),
    ], additional_targets={'dsm': 'image'})

    pixel_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    return spatial_transform, pixel_transform

def get_validation_transforms(height=256, width=256):
    spatial_transform = A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(height=height, width=width),
    ], additional_targets={'dsm': 'image'}) 

    pixel_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    return spatial_transform, pixel_transform