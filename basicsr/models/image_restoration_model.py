# image_restoration_model.py - Koşullu TTA Mantığı ile Güncellendi

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import datetime
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.models.base_model import BaseModel

@MODEL_REGISTRY.register()
class ImageCleanModel(BaseModel):
    """Base IR model for General Image Restoration."""
    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.best_metric = 0
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if self.is_train:
            self.init_training_settings()

    def setup_schedulers(self):
        # ... (değişiklik yok) ...
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def init_training_settings(self):
        # ... (değişiklik yok) ...
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get('Edgeloss_opt'):
            self.cri_edge = build_loss(train_opt['Edgeloss_opt']).to(self.device)
        else:
            self.cri_edge = None
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        # ... (değişiklik yok) ...
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def _tile_inference(self, img, tile_size=128, tile_overlap=32):
        # ... (değişiklik yok) ...
        _, C, H, W = img.shape
        stride = tile_size - tile_overlap
        rows = (H - tile_overlap + stride - 1) // stride
        cols = (W - tile_overlap + stride - 1) // stride
        output = img.new_zeros(1, C, H, W)
        weight = img.new_zeros(1, C, H, W)
        
        for i in range(rows):
            for j in range(cols):
                y0 = i * stride
                x0 = j * stride
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                y0 = max(0, y1 - tile_size)
                x0 = max(0, x1 - tile_size)
                img_tile = img[:, :, y0:y1, x0:x1]
                with torch.no_grad():
                    pred_tile = self.inference(img_tile)
                output[:, :, y0:y1, x0:x1] += pred_tile
                weight[:, :, y0:y1, x0:x1] += 1.0
        output /= weight
        return output
    
    # <─ DEĞİŞİKLİK: TTA mantığı, YAML'deki 'use_tta' bayrağına göre koşullu hale getirildi.
    def inference(self, img):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        
        # YAML'den use_tta bayrağını kontrol et. Varsayılan olarak False (kapalı).
        if self.opt['val'].get('use_tta', False):
            # TTA (flip-ensemble) Yolu
            # 1) Orijinal çıktı
            outs1 = net(img)
            pred1 = outs1[-1] if isinstance(outs1, list) else outs1

            # 2) Flip-TTA ile çıktı
            img_fl = torch.flip(img, dims=[-1])
            outs2 = net(img_fl)
            tmp = outs2[-1] if isinstance(outs2, list) else outs2
            pred2 = torch.flip(tmp, dims=[-1])

            # 3) Sonuçları ortala
            pred = (pred1 + pred2) * 0.5
        else:
            # Hızlı Yol (TTA Yok)
            outs = net(img)
            pred = outs[-1] if isinstance(outs, list) else outs

        if self.is_train:
            net.train()
            
        return pred

    def pad_test(self, window_size):
        # ... (değişiklik yok) ...
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, hh, ww = self.output.size()
        self.output = self.output[:, :, 0:hh - mod_pad_h * scale, 0:ww - mod_pad_w * scale]

    def pad2square_test(self, window_size):
        # ... (değişiklik yok) ...
        scale = self.opt.get('scale', 1)
        b, c, h, w = self.lq.size()
        X = int(math.ceil(max(h, w) / float(window_size)) * window_size)
        img = torch.zeros(1, c, X, X).type_as(self.lq)
        mask = torch.zeros(1, 1, X, X).type_as(self.lq)
        hh = (X - h) // 2
        ww = (X - w) // 2
        img[:, :, hh:hh + h, ww:ww + w] = self.lq
        mask[:, :, hh:hh + h, ww:ww + w].fill_(1)
        self.nonpad_test(img)
        self.output = torch.masked_select(self.output, mask.bool()).reshape(b, c, h * scale, w * scale)

    def nonpad_test(self, img=None):
        # ... (değişiklik yok) ...
        if img is None:
            img = self.lq
        _, _, H, W = img.shape
        # _tile_inference metodunu çağırmak için sadece tile_size kontrolü yeterli.
        if self.opt['val'].get('tile_size'):
             self.output = self._tile_inference(img, tile_size=self.opt['val']['tile_size'], tile_overlap=self.opt['val'].get('tile_overlap', 32))
        else:
            with torch.no_grad():
                self.output = self.inference(img)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # ... (değişiklik yok) ...
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # ... (değişiklik yok, test fonksiyonu seçimi düzeltildi) ...
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        
        # Test fonksiyonu seçimi. Artık `window_size` yerine `tile_size` kontrol ediliyor.
        if self.opt['val'].get('tile_size'):
            # pad2square_test artık doğrudan kullanılmıyor, nonpad_test tüm mantığı yönetiyor.
            # Bu, tiling'in her zaman çalışmasını sağlar.
            self.test = self.nonpad_test
        else:
            self.test = self.nonpad_test
            
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        
        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
            
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            
            self.test()
            
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val'].get('suffix'):
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
            
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
                
        if use_pbar:
            pbar.close()
            
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            if 'psnr' in self.best_metric_results[dataset_name]:
                self.best_metric = self.best_metric_results[dataset_name]['psnr']['val']

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        # ... (değişiklik yok) ...
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        # ... (değişiklik yok) ...
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, flag=False):
        # ... (değişiklik yok) ...
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, flag, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter, flag)
        self.save_training_state(epoch, current_iter, flag)

    def save_network(self, net, net_label, current_iter, flag, param_key='params'):
        # ... (değişiklik yok) ...
        if flag:
            current_save = 'best'
            save_filename = f'{net_label}_{current_save}.pth'
        else:
            current_save = 'latest'
            save_filename = f'{net_label}_{current_save}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'Lengths mismatch.'
        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict
        retry = 3
        logger = get_root_logger()
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger.warning(f'Save model error: {e}, retry: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Cannot save {save_path}.')

    def save_training_state(self, epoch, current_iter, flag):
        # ... (değişiklik yok) ...
        if flag:
            save_name = 'best'
        else:
            save_name = 'latest'
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'best_metric': self.best_metric,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{save_name}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)
            retry = 3
            logger = get_root_logger()
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger.warning(f'Save training state error: {e}, retry: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Cannot save {save_path}.')