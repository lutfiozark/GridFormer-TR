# basicsr/losses/__init__.py (Güncellenmiş Hali)

from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty)
from .hdr_charbonnier_loss import HDRCharbonnierLoss
from .msssim_loss import MSSSIMLoss     # <─ YENİ
from .edge_loss    import EdgeLoss       # <─ YENİ


__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize',
    'HDRCharbonnierLoss',
    'MSSSIMLoss', # <─ YENİ
    'EdgeLoss'    # <─ YENİ
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt_copy = deepcopy(opt)
    loss_type = opt_copy.pop('type')
    
    loss_cls = LOSS_REGISTRY.get(loss_type)
    if loss_cls is None:
        raise ValueError(f"Loss type '{loss_type}' is not registered. Available losses: {list(LOSS_REGISTRY.keys())}")

    loss_instance = loss_cls(**opt_copy)
    
    logger = get_root_logger()
    logger.info(f'Loss [{loss_instance.__class__.__name__}] is created.')
    return loss_instance