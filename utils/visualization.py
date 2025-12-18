import wandb
import torch

def log_predictions(images, masks, preds):
    images = images.cpu()
    masks = masks.cpu()
    preds = torch.argmax(preds, dim=1).cpu()

    wandb.log({
        "examples": [
            wandb.Image(
                img.permute(1, 2, 0).numpy(),
                masks={
                    "ground_truth": {"mask_data": mask.numpy()},
                    "prediction": {"mask_data": pred.numpy()}
                }
            )
            for img, mask, pred in zip(images, masks, preds)
        ]
    })
