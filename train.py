import torch
import wandb
import json

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.trainer import Trainer


from datasets.RGBSARdataset import RGB_SAR_Dataset
from datasets.transforms_sar import get_transforms, get_validation_transforms

from training.losses import MultiTaskLoss
# from models.unet_sar import MultiTaskNetSar
# from models.unet_dual_sar import MultiTaskDualUnet
# from models.unet import MultiTaskUnet
from models.unet_sar_only import MultiTaskUnetSarOnly

# from models.deeplab import MultiTaskDeepLabV3
# from models.deeplab_sar import MultiTaskDeepLabV3Sar


def main():

    with open("config.json") as cfg:
        config = json.load(cfg)

    wandb.init(project=config['project_name'])

    mdl_config = config['model_config']
    mdl_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_config['max_height'] = config['max_height']

    model = MultiTaskUnetSarOnly()
    model.to(mdl_config['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=mdl_config["lr"])

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )

    loss_fn = MultiTaskLoss(seg_weight=1, dsm_weight=100)

    train_dataset = RGB_SAR_Dataset(
        json_data_file=config['metadata_train_file'],
        img_dir=config['train_images'],
        dsm_dir=config['train_dsm'],
        sar_dir=config['train_sar'],
        transfrom_getter=get_transforms
    )

    val_dataset = RGB_SAR_Dataset(
        json_data_file=config['metadata_valid_file'],
        img_dir=config['valid_images'],
        dsm_dir=config['valid_dsm'],
        sar_dir=config['valid_sar'],
        transfrom_getter=get_validation_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=mdl_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=mdl_config["batch_size"])

    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, mdl_config, scheduler, sar_only=True)

    best_result = 0.0
    for epoch in range(mdl_config["epochs"]):
        trainer.train_epoch(epoch)
        _, results = trainer.validate(epoch)

        if results['Final_Score'] > best_result:
            best_result = results['Final_Score']
            torch.save(model.state_dict(), config['best_model_path'])
            wandb.save(config['best_model_path'])

if __name__ == "__main__":
    main()
