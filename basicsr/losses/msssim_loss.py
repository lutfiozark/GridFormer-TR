# basicsr/losses/ms_ssim_loss.py (Güncellenmiş Hali)

import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

try:
    from pytorch_msssim import ms_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False
    print("Warning: pytorch-msssim not available. Install with: pip install pytorch-msssim")


@LOSS_REGISTRY.register()
class MSSSIMLoss(nn.Module):
    """
    MS-SSIM loss implemented using the pytorch-msssim library.
    The loss is calculated as 1 - MS-SSIM.

    For small images (< 160x160), falls back to L1 loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        window_size (int): Window size for SSIM computation. Default: 11.
        sigma (float): Standard deviation for Gaussian window. Default: 1.5.
        data_range (float): Data range for SSIM computation. Default: 1.0.
        size_average (bool): Whether to average the result. Default: True.
        min_size (int): Minimum image size for MS-SSIM. Default: 160.
    """

    def __init__(self, loss_weight=1.0, window_size=11, sigma=1.5,
                 data_range=1.0, size_average=True, min_size=160, **kwargs):
        super(MSSSIMLoss, self).__init__()

        if not MSSSIM_AVAILABLE:
            raise ImportError(
                "pytorch-msssim is required for MSSSIMLoss. Install with: pip install pytorch-msssim")

        self.loss_weight = loss_weight
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.min_size = min_size

        # **kwargs'ı ignore et (config'den gelebilecek bilinmeyen parametreler için)
        print(
            f"MSSSIMLoss initialized: weight={loss_weight}, min_size={min_size}")

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted tensor with shape (N, C, H, W), range [0, 1].
            target (Tensor): Target tensor with shape (N, C, H, W), range [0, 1].
        Returns:
            Tensor: Calculated MS-SSIM loss.
        """
        if not MSSSIM_AVAILABLE:
            raise RuntimeError("pytorch-msssim is not available")

        # Tensörlerin device'ını kontrol et
        if pred.device != target.device:
            target = target.to(pred.device)

        # Boyut kontrolü
        B, C, H, W = pred.shape
        target_H, target_W = target.shape[2], target.shape[3]

        # Eğer prediction ve target boyutları farklıysa, target'ı resize et
        if H != target_H or W != target_W:
            target = F.interpolate(target, size=(
                H, W), mode='bilinear', align_corners=False)

        # MS-SSIM minimum boyut kontrolü
        if min(H, W) < self.min_size:
            # Küçük görüntüler için L1 loss kullan
            l1_loss = F.l1_loss(pred, target)
            return self.loss_weight * l1_loss

        # MS-SSIM hesaplama
        try:
            # En basit kullanım - sadece gerekli parametreler
            ms_ssim_val = ms_ssim(
                pred, target,
                data_range=self.data_range,
                size_average=self.size_average
            )

            # SSIM loss = 1 - SSIM (çünkü SSIM yüksek olması iyi, loss düşük olması iyi)
            loss_val = 1.0 - ms_ssim_val
            return self.loss_weight * loss_val

        except Exception as e:
            # Fallback: L1 loss
            print(f"MS-SSIM calculation failed, using L1 loss: {e}")
            l1_loss = F.l1_loss(pred, target)
            return self.loss_weight * l1_loss
