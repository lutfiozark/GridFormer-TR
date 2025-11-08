from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.models import lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch

@MODEL_REGISTRY.register()
class MIMOModel(ImageCleanModel):
    """Base IR model for General Image Restoration."""
    
    def optimize_parameters(self, current_iter, scaler):
        self.optimizer_g.zero_grad()
        # AMP: otomatik karışık hassasiyet modunda ileri geçiş
        with torch.amp.autocast(device_type='cuda'):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]
            self.output = preds[-1]
            l_total = 0.
            loss_dict = OrderedDict()

            # Multi-image ground truth oluşturma:
            gt_images = [0 for _ in range(len(preds))]
            gt_images[0] = self.gt
            for i in range(1, len(preds)):
                gt_images[i] = F.interpolate(gt_images[i - 1], scale_factor=0.5, mode='bilinear', recompute_scale_factor=True)
            gt_images.reverse()

            # Pixel loss
            if self.cri_pix:
                l_pix = 0.
                for j in range(len(preds)):
                    l_pix += self.cri_pix(preds[j], gt_images[j])
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # Perceptual loss
            if self.cri_perceptual:
                l_percep = 0.
                for i in range(len(preds)):
                    l_percep1, _ = self.cri_perceptual(preds[i], gt_images[i])
                    l_percep += l_percep1
                l_total += l_percep
                loss_dict['l_percep'] = l_percep

            # Edge loss (varsa)
            if self.cri_edge:
                l_edge = 0.
                for pred in preds:
                    l_edge += self.cri_edge(pred, self.gt)
                l_total += l_edge
                loss_dict['l_edge'] = l_edge

        scaler.scale(l_total).backward()
        scaler.step(self.optimizer_g)
        scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

print("MIMOModel başarıyla kaydedildi!")
