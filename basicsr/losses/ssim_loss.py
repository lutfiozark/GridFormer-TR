import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.util.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """SSIM (Structural Similarity) Loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction method: 'none' | 'mean' | 'sum'.
            Default: 'mean'.
        window_size (int): Window size for SSIM calculation. Default: 11.
        val_range (float): Maximum value range of input images. Default: 1.0.
        channel (int): Number of channels. Default: 3.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', window_size=11, val_range=1.0, channel=3):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.window_size = window_size
        self.val_range = val_range
        self.channel = channel

        # Create a 2D Gaussian window
        self.window = self._create_window(window_size, channel)
        self.window = nn.Parameter(self.window, requires_grad=False)

    def _create_window(self, window_size, channel):
        """Create a 2D Gaussian window.

        Args:
            window_size (int): Window size.
            channel (int): Number of channels.

        Returns:
            Tensor: Created window.
        """
        def gaussian(window_size, sigma):
            # Create position indices
            indices = torch.arange(window_size, dtype=torch.float)
            # Calculate distances from center
            centered_indices = indices - window_size // 2
            # Calculate Gaussian values
            gaussian_values = torch.exp(-(centered_indices.pow(2)
                                          ) / (2 * sigma**2))
            # Normalize
            return gaussian_values / gaussian_values.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).
            y (Tensor): Target tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        (_, channel, height, width) = x.size()

        if channel == self.channel and self.window.device == x.device:
            window = self.window
        else:
            window = self._create_window(
                self.window_size, channel).to(x.device)
            window = nn.Parameter(window, requires_grad=False)

        # Calculate SSIM
        ssim_value = self._ssim(
            x, y, window, self.window_size, channel, self.val_range)
        loss = 1.0 - ssim_value  # SSIM loss (1 - SSIM)

        return self.loss_weight * loss

    def _ssim(self, img1, img2, window, window_size, channel, val_range=1.0):
        """Calculate SSIM.

        Args:
            img1 (Tensor): Input tensor.
            img2 (Tensor): Target tensor.
            window (Tensor): Gaussian window.
            window_size (int): Window size.
            channel (int): Number of channels.
            val_range (float): Maximum value range of input images.

        Returns:
            Tensor: SSIM value.
        """
        # Constants for stability
        C1 = (0.01 * val_range)**2
        C2 = (0.03 * val_range)**2

        # Calculate mean
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        # Calculate squared mean
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variance and covariance
        sigma1_sq = F.conv2d(img1 * img1, window,
                             padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window,
                             padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window,
                           padding=window_size // 2, groups=channel) - mu1_mu2

        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
            ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Return mean SSIM
        if self.reduction == 'mean':
            return ssim_map.mean()
        elif self.reduction == 'sum':
            return ssim_map.sum()
        else:  # 'none'
            return ssim_map
