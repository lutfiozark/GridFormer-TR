# basicsr/losses/tv_loss.py

import torch
import torch.nn as nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    """Total Variation Loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction method: 'none' | 'mean' | 'sum'.
            Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TVLoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported modes are "mean", "sum", "none".')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x, gt=None): # gt parametresi uyumluluk için eklendi ancak kullanılmıyor.
        """
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).
            gt (Tensor, optional): Ground-truth tensor. Not used in TVLoss.

        Returns:
            Tensor: Forward results.
        """
        b, c, h, w = x.size()

        # Yatay varyasyon
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
        # Dikey varyasyon
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)

        if self.reduction == 'mean':
            return self.loss_weight * (torch.sum(tv_h) + torch.sum(tv_w)) / (b * c * h * w)
        elif self.reduction == 'sum':
            return self.loss_weight * (torch.sum(tv_h) + torch.sum(tv_w))
        else: # 'none'
            return self.loss_weight * (tv_h + tv_w)