import torch


def blackout(imgs_tnsr):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        returns torch.tensor 
    """
    return torch.ones(imgs_tnsr.shape) * 255
