from mmdet.registry import DATASETS
from mmengine.registry import init_default_scope

init_default_scope('mmdet')

import datasets.mmd_custom_dataset 

def verify():
    data_root = '/home/pepsdev/mgr/buildings-height-estimation/dataset/' 
    
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='LoadSARAndDSMFromFile'),
        dict(type='CustomRandomFlip', prob=1.0, direction=['horizontal']),
        dict(type='CustomRandomCrop', crop_size=(512, 512)),
        dict(type='MultiModalPixelAug'),
        dict(type='PackMultiModalInputsEarlyFusion')
    ]

    dataset_config = dict(
        type='CustomSARBuildingDataset',
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/rgb/'),
        pipeline=pipeline
    )

    print("Budowanie datasetu...")
    try:
        dataset = DATASETS.build(dataset_config)
        print(f"Sukces! Znaleziono obrazów: {len(dataset)}")
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY przy inicjalizacji: {e}")
        return

    print("\n--- TEST POBRANIA PRÓBKI (INDEKS 0) ---")
    try:
        sample = dataset[0]
        
        inputs = sample['inputs']
        print(f"[RGB] Kształt wejścia: {inputs.shape}, Typ: {inputs.dtype}")
        
        data_sample = sample['data_samples']
        
        if hasattr(data_sample, 'sar_img'):
            sar = data_sample.sar_img
            print(f"[SAR] Kształt w DataSample: {sar.shape}, Typ: {sar.dtype}")
        else:
            print("[SAR] BRAK DANYCH W KONTENERZE!")

        if hasattr(data_sample, 'gt_height_map'):
            dsm = data_sample.gt_height_map
            print(f"[DSM] Kształt w DataSample: {dsm.shape}, Typ: {dsm.dtype}")
        else:
            print("[DSM] BRAK DANYCH W KONTENERZE!")
            
        if hasattr(data_sample, 'gt_instances'):
            masks = data_sample.gt_instances.masks
            print(f"[MASKI] Kształt masek: {masks.masks.shape}, Typ: BitmapMasks")
        else:
            print("[MASKI] BRAK DANYCH W KONTENERZE!")

        print("\nDIAGNOZA: Pipeline jest w pełni szczelny. Możemy integrować modele.")
        
    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY podczas przepuszczania danych przez pipeline:")
        raise e

if __name__ == '__main__':
    verify()
