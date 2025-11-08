# basicsr/losses/edge_loss.py (Güncellenmiş Hali)

import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    """
    Sobel edge loss based on L1 difference of gradient maps.
    """
    def __init__(self, loss_weight=1.0):
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        # Define Sobel filters
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        # Register kernels as buffers to move them to the correct device automatically
        self.register_buffer('kx', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('ky', sobel_y.view(1, 1, 3, 3))

    def _get_gradient(self, img):
        """Calculates the gradient map of an image using Sobel filters."""
        # Use groups=img.shape[1] to apply the same kernel to each channel independently
        gx = F.conv2d(img, self.kx.repeat(img.shape[1], 1, 1, 1), padding=1, groups=img.shape[1])
        gy = F.conv2d(img, self.ky.repeat(img.shape[1], 1, 1, 1), padding=1, groups=img.shape[1])
        # Calculate gradient magnitude, add epsilon for stability
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, pred, target):
        """
        Calculates L1 loss between the edge maps of prediction and ground-truth.
        """
        loss = F.l1_loss(self._get_gradient(pred), self._get_gradient(target))
        return self.loss_weight * loss