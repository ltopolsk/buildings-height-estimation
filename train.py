import torch
import wandb
import json

from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet, DeepLabV3

from training.trainer import Trainer
from datasets.DFCdataset import BasicDataSet

def main():

    with open("config.json") as cfg:
        config = json.load(cfg)

    wandb.init(project=config['project_name'])

    mdl_config = config['model_config']
    mdl_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    #TODO: implement choice of model architectures
    model = Unet(classes=mdl_config["num_classes"]).to(mdl_config["device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=mdl_config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataset = BasicDataSet(config['metadata_train_file'], img_dir=config['train_images'])
    val_dataset = BasicDataSet(config['metadata_valid_file'], img_dir=config['valid_images'])

    train_loader = DataLoader(train_dataset, batch_size=mdl_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=mdl_config["batch_size"])

    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, mdl_config)

    best_iou = 0.0
    for epoch in range(mdl_config["epochs"]):
        trainer.train_epoch(epoch)
        val_loss, val_iou = trainer.validate(epoch)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config['best_model_path'])
            wandb.save(config['best_model_path'])


if __name__ == "__main__":
    main()
