# basicsr/losses/hdr_charbonnier_loss.py

import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class HDRCharbonnierLoss(nn.Module):
    """γ-Charbonnier (HDR-friendly) loss."""
    def __init__(self, loss_weight=1.0, reduction='mean', gamma=0.6, eps=1e-3):
        super().__init__()
        self.w, self.red, self.g, self.eps = loss_weight, reduction, gamma, eps
        
    def forward(self, pred, target):
        diff = (pred - target).abs() + self.eps
        loss = torch.pow(diff, self.g)
        if self.red == 'mean':
            loss = loss.mean()
        elif self.red == 'sum':
            loss = loss.sum()
        return self.w * loss