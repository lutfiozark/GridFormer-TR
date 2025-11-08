import importlib
from copy import deepcopy
from os import path as osp
import os
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(file)[0] for file in os.listdir(model_folder) if file.endswith('_model.py')]
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]

def build_model(opt):
    opt = deepcopy(opt)
    model_type = opt['model_type']
    net_opt = opt.get('network_g', {})
    logger = get_root_logger()

    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Model type '{model_type}' not found in registry.")

    model = model_cls(opt)
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    return model