import torch

def dice_score(preds, targets, epsilon=1e-6):
    """
    preds: torch.Tensor (N, 1, H, W) - binary mask (0 or 1)
    targets: torch.Tensor (N, 1, H, W) - binary ground truth mask
    """
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean()