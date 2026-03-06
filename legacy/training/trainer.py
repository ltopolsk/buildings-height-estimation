import torch
import wandb
from training.metrics import SegHeightMetrics
from utils.visualization import log_predictions


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, config, scheduler=None, sar_only=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config["device"]
        self.max_building_height = config["max_height"]
        self.scheduler = scheduler
        self.val_metrics = SegHeightMetrics(device=config["device"])
        self.sar_only  = sar_only

    def train_epoch(self, epoch):
        self.model.train()

        running_total_loss = 0.0
        running_seg_loss = 0.0
        running_dsm_loss = 0.0

        n = len(self.train_loader)

        for items in self.train_loader:

            for key, item in items.items():
                items[key] = item.to(self.device)

            seg_pred, dsm_pred = self.model(items)
            step_loss, step_seg_loss, step_dsm_loss = self.loss_fn(seg_pred, dsm_pred, items['mask'], items['dsm'])

            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()

            running_total_loss += step_loss.item()
            running_seg_loss += step_seg_loss.item()
            running_dsm_loss += step_dsm_loss.item()

            wandb.log({
                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                "train/step_loss": step_loss.item(), 
                "train/step_seg_loss": step_seg_loss.item(),
                "train/step_dsm_loss": step_dsm_loss.item()
            })
        
        avg_loss = running_total_loss / n
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/seg_loss": running_seg_loss / n,
            "train/dsm_loss": running_dsm_loss / n,
            "epoch": epoch
        })

        return avg_loss


    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        for i, items in enumerate(self.val_loader):

            for key, item in items.items():
                items[key] = item.to(self.device)

            seg_pred, dsm_pred = self.model(items)


            loss, _, _ = self.loss_fn(seg_pred, dsm_pred, items['mask'], items['dsm'])
            total_loss += loss.item()
    
            pred_mask_binary = (torch.sigmoid(seg_pred) > 0.5).float()

            # denormalize dsm_pred and dsms (new unit -> meters)
            dsms_meters = items['dsm']* self.max_building_height
            dsm_pred_meters = dsm_pred * self.max_building_height

            if i == 0:
                log_predictions(
                    rgb=items['rgb'],
                    sar=items['sar'],
                    mask_gt=items['mask'],
                    mask_pred=pred_mask_binary,
                    dsm_gt=dsms_meters,
                    dsm_pred=dsm_pred_meters,\
                    max_height=self.max_building_height
                )

            self.val_metrics.update(dsm_pred_meters, dsms_meters, seg_pred, items['mask'])

        avg_loss = total_loss / len(self.val_loader)
        results = self.val_metrics.compute()

        wandb.log({
            "val/loss": avg_loss,
            "val/AP50": results['AP50'],
            "val/Delta1": results['Delta1'],
            "epoch": epoch
        })

        self.val_metrics.reset()

        if self.scheduler:
            self.scheduler.step(avg_loss)

        return avg_loss, results
