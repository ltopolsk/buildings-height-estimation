import torch
import wandb
from training.metrics import iou_score
from utils.visualization import log_predictions


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config["device"]
        self.num_classes = config["num_classes"]


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            preds = self.model(images)
            loss = self.loss_fn(preds, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            wandb.log({"train/loss": loss.item(), "epoch": epoch})

        return total_loss / len(self.train_loader)


    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0

        for i, (images, masks) in enumerate(self.val_loader):

            images = images.to(self.device)
            masks = masks.to(self.device)
            preds = self.model(images)
            if i%10 == 0:
                log_predictions(images, masks, preds)
            
            loss = self.loss_fn(preds, masks)

            total_loss += loss.item()
            total_iou += iou_score(preds, masks, self.num_classes)

        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)

        wandb.log({
            "val/loss": avg_loss,
            "val/iou": avg_iou,
            "epoch": epoch
        })

        return avg_loss, avg_iou
