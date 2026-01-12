import torch
import wandb
import json

from torch.utils.data import DataLoader
from training.trainer import Trainer
from datasets.RGBSARdataset import RGB_SAR_Dataset
from datasets.transforms_sar import get_validation_transforms
from training.losses import MultiTaskLoss
from utils.late_fusion import WeightedFusionWrapper

from models.unet import MultiTaskUnet
from models.unet_sar_only import MultiTaskUnetSarOnly

def main():

    with open("config.json") as cfg:
        config = json.load(cfg)

    wandb.init(project=config['project_name'])

    mdl_config = config['model_config']
    mdl_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_config['max_height'] = config['max_height']

    rgb_model = MultiTaskUnet(encoder_name='resnet34').to(mdl_config['device'])
    rgb_model.load_state_dict(torch.load("best_models/best_model_unet_rgb.pth"))

    sar_model = MultiTaskUnetSarOnly(encoder_name='resnet34').to(mdl_config['device'])
    sar_model.load_state_dict(torch.load("best_models/best_model_unet_sar.pth"))
    fusion_model = WeightedFusionWrapper(
        rgb_model, 
        sar_model, 
        alpha_seg=0.0, 
        alpha_dsm=0.3  
    ).to(mdl_config['device'])

    loss_fn = MultiTaskLoss(seg_weight=1, dsm_weight=100)

    val_dataset = RGB_SAR_Dataset(
        json_data_file=config['metadata_valid_file'],
        img_dir=config['valid_images'],
        dsm_dir=config['valid_dsm'],
        sar_dir=config['valid_sar'],
        transfrom_getter=get_validation_transforms
    )

    val_loader = DataLoader(val_dataset, batch_size=mdl_config["batch_size"])

    trainer = Trainer(
        model=fusion_model,
        optimizer=None,
        loss_fn=loss_fn,
        train_loader=None,
        val_loader=val_loader,
        config=mdl_config
    )

    loss, metrics = trainer.validate(epoch=0)
    print(f"Decision Fusion Results: {metrics}")
    print(f"Finall loss: {loss}")

if __name__ == "__main__":
    main()
