# basicsr/losses/dists_loss.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger

__all__ = ['DISTS']


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        self.filter_size = filter_size
        a = torch.ones(filter_size, filter_size).float()
        a = a / a.sum().sqrt()
        a = a.unsqueeze(0).unsqueeze(0)
        self.register_buffer('filter', a.repeat(self.channels, 1, 1, 1))

    def forward(self, input):
        k = self.filter_size
        pad = self.padding
        h, w = input.size(2), input.size(3)

        # Eğer o kadar küçük ki reflect pad hata veriyor, pad'i clamp edelim
        pad_h = max(0, k - h)
        pad_w = max(0, k - w)

        # reflect pad için her pad < ilgili dim olmalı
        if pad_h >= h:
            pad_h = h - 1 if h > 1 else 0
        if pad_w >= w:
            pad_w = w - 1 if w > 1 else 0

        if pad_h > 0 or pad_w > 0:
            ph1, ph2 = pad_h // 2, pad_h - pad_h // 2
            pw1, pw2 = pad_w // 2, pad_w - pad_w // 2
            # (left, right, top, bottom)
            input = F.pad(input, (pw1, pw2, ph1, ph2), mode='reflect')

        # 1×1 gibi çok küçük hale geldiyse normal L2 pool atlamak yerine
        # fallback olarak avg pool uygulayalım (etkisi yumuşar ama kalır)
        if input.size(2) < k or input.size(3) < k:
            return F.avg_pool2d(input, kernel_size=1, stride=self.stride)

        # Şimdi çekirdek her zaman sığacak; tam L2-pooling uygula
        return torch.sqrt(
            F.conv2d(
                input**2, self.filter, stride=self.stride, padding=pad, groups=input.shape[1]
            ) + 1e-6)


@LOSS_REGISTRY.register()
class DISTS(nn.Module):
    """DISTS (Deep Image Structure and Texture Similarity) metric.

    References:
        Ding et al. Image Quality Assessment: Unifying Structure and Texture Similarity.
        TPAMI 2020.
        https://github.com/dingkeyan93/DISTS

    Args:
        loss_weight (float): Loss weight for DISTS metric. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        calibrated (bool): Whether to use calibrated version. Default: True.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', calibrated=True):
        super(DISTS, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.calibrated = calibrated
        self.logger = get_root_logger()

        # Check if weights.pt exists
        weights_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'weights.pt')

        self.register_buffer('alpha', torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]))
        self.register_buffer('beta', torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))

        if self.calibrated:
            self.alpha.data = torch.tensor(
                [0.1680, 0.1018, 0.0416, 0.0397, 0.0397])
            self.beta.data = torch.tensor(
                [0.0316, 0.3153, 0.2303, 0.2085, 0.2086])

        # Initialize VGG model
        try:
            vgg = models.vgg16(weights=None)
            # Try to load DISTS weights
            if not os.path.exists(weights_path):
                self.logger.warning(
                    f'DISTS weights file not found at {weights_path}. '
                    'Falling back to ImageNet pretrained weights.'
                )
                # Use pretrained ImageNet weights if DISTS weights not found
                vgg = models.vgg16(pretrained=True)
            else:
                try:
                    vgg.load_state_dict(torch.load(weights_path))
                    self.logger.info(
                        f"DISTS weights loaded from {weights_path}.")
                except Exception as e:
                    self.logger.warning(
                        f"Could not load DISTS weights from {weights_path}: {e}. "
                        "Falling back to ImageNet pretrained weights."
                    )
                    vgg = models.vgg16(pretrained=True)
        except Exception as e:
            self.logger.warning(f"Error initializing VGG model: {e}")
            # Fallback to newer torchvision API if available
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except:
                vgg = models.vgg16(pretrained=True)

        # Extract VGG feature layers
        self.vgg_layers = nn.ModuleList([
            nn.Sequential(*list(vgg.features.children())[:4]),  # relu1_2
            nn.Sequential(*list(vgg.features.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.features.children())[9:16]),  # relu3_3
            nn.Sequential(*list(vgg.features.children())[16:23]),  # relu4_3
            nn.Sequential(*list(vgg.features.children())[23:30]),  # relu5_3z<
        ])

        # L2 pooling for different channel counts
        self.L2_pooling = nn.ModuleList([
            L2pooling(channels=64),   # For relu1_2
            L2pooling(channels=128),  # For relu2_2
            L2pooling(channels=256),  # For relu3_3
            L2pooling(channels=512)   # For relu4_3
        ])

        # Freeze the VGG network
        for param in self.parameters():
            param.requires_grad = False

        # MaxPool2d katmanlarını ceil_mode=True olacak şekilde güncelle
        for vgg_module in self.vgg_layers:
            for layer in vgg_module:
                if isinstance(layer, nn.MaxPool2d):
                    layer.ceil_mode = True
                elif isinstance(layer, nn.Sequential):
                    for sublayer in layer:
                        if isinstance(sublayer, nn.MaxPool2d):
                            sublayer.ceil_mode = True

    def forward_once(self, x):
        """Forward function for single image.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            list: List of feature maps at different VGG layers.
        """
        feats = []
        for i, module in enumerate(self.vgg_layers):
            x = module(x)
            feats.append(x)

            # Pooling after every step except the last
            if i < len(self.vgg_layers) - 1:
                x = self.L2_pooling[i](x)

        return feats

    def compute_similarity(self, feat_x, feat_y):
        """Compute structure and texture similarity.

        Args:
            feat_x (list): Features from image x
            feat_y (list): Features from image y

        Returns:
            tuple: Structure similarity and texture similarity
        """
        struct_sims = []
        text_sims = []

        for fx, fy in zip(feat_x, feat_y):
            # compute stats per feature map
            mu_x = fx.mean(dim=[2, 3], keepdim=True)
            mu_y = fy.mean(dim=[2, 3], keepdim=True)
            sigma_x = fx.var(dim=[2, 3], unbiased=False, keepdim=True)
            sigma_y = fy.var(dim=[2, 3], unbiased=False, keepdim=True)
            cov_xy = ((fx - mu_x) * (fy - mu_y)).mean(dim=[2, 3], keepdim=True)

            # structural similarity
            c1 = 1e-4
            struct = (2 * mu_x * mu_y + c1) / (mu_x.pow(2) + mu_y.pow(2) + c1)

            # textural similarity
            c2 = 1e-3
            text = (2 * cov_xy + c2) / (sigma_x + sigma_y + c2)

            # flatten per-channel
            struct = struct.view(fx.size(0), -1)  # [batch, C]
            text = text.view(fx.size(0), -1)

            # average across channels
            struct_sims.append(struct.mean(dim=1))  # [batch]
            text_sims.append(text.mean(dim=1))      # [batch]

        return struct_sims, text_sims

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).
            y (Tensor): Ground-truth tensor with shape (N, C, H, W).

        Returns:
            Tensor: DISTS loss.
        """
        # Rescale to [0, 1] if needed
        if torch.max(x) > 1.0 or torch.min(x) < 0.0:
            x = (x + 1) / 2.0  # Assuming [-1, 1] range
        if torch.max(y) > 1.0 or torch.min(y) < 0.0:
            y = (y + 1) / 2.0  # Assuming [-1, 1] range

        # Extract features
        feat_x = self.forward_once(x)
        feat_y = self.forward_once(y)

        # Compute similarity
        struct_sims, text_sims = self.compute_similarity(feat_x, feat_y)

        # combine per-level sims
        losses = []  # list of [batch] tensors
        for i, (s, t) in enumerate(zip(struct_sims, text_sims)):
            loss_i = self.alpha[i] * (1 - s) + self.beta[i] * (1 - t)
            losses.append(loss_i)

        # stack over levels -> [levels, batch]
        loss_levels = torch.stack(losses, dim=0)

        # average over levels -> [batch]
        loss_batch = loss_levels.mean(dim=0)

        # final reduction
        if self.reduction == 'mean':
            return self.loss_weight * loss_batch.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss_batch.sum()
        else:
            return self.loss_weight * loss_batch


@LOSS_REGISTRY.register()
class DISTSLoss(DISTS):
    """DISTS loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DISTSLoss, self).__init__(
            loss_weight, reduction, calibrated=True)
