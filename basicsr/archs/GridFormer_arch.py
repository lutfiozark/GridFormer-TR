"""
GridFormer_arch.py - Son Versiyon (Cross-Attention Yumuşatması ve diğer tüm iyileştirmelerle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
import torch.utils.checkpoint as cp
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

# ... (Yardımcı Fonksiyonlar, SEBlock, LayerNorm, DropPath, FeedForward, LocalAttention, CrossAttention, DynamicFusion sınıfları aynı kalıyor) ...

##############################################################################
## Yardımcı Fonksiyonlar: to_3d ve to_4d

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##############################################################################
## SEBlock Sınıfı
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False), 
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

##############################################################################
## Layer Norm Sınıfları

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_3d = to_3d(x)
        out_3d = self.body(x_3d)
        out_4d = to_4d(out_3d, h, w)
        return out_4d

##############################################################################
## DropPath Sınıfı

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,)*(x.ndim-1)).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask

##############################################################################
## FeedForward

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##############################################################################
## Eski Attention Modülleri (LocalAttention)

class LocalAttention(nn.Module):
    def __init__(self, dim, sample_rate=1, bias=False, enable_fs=False):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.sr = sample_rate
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        c_ = self.dim
        q, k, v = torch.split(qkv, c_, dim=1)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.einsum('b c n, b c m -> b n m', q, k)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('b n m, b c m -> b c n', attn, v)
        out = rearrange(out, 'b c (h w) -> b c h w', h=H, w=W)
        out = self.proj_out(out)
        return out

##############################################################################
## Yeni Eklenen Modüller: CrossAttention, DynamicFusion, InnovativeMultiScaleCrossAttention

class CrossAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.key_conv   = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_out   = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.scale = dim ** -0.5

    def forward(self, x_query, x_key_value):
        query = self.query_conv(x_query)
        key   = self.key_conv(x_key_value)
        value = self.value_conv(x_key_value)
        B, C, H, W = query.shape
        query = query.view(B, C, -1).permute(0, 2, 1)
        key   = key.view(B, C, -1)
        value = value.view(B, C, -1).permute(0, 2, 1)
        attn = torch.bmm(query, key) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out  = torch.bmm(attn, value)
        out  = out.permute(0, 2, 1).view(B, C, H, W)
        out  = self.proj_out(out)
        return out

class DynamicFusion(nn.Module):
    def __init__(self, in_channels):
        super(DynamicFusion, self).__init__()
        self.conv_gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        if self.conv_gate.bias is not None:
            self.conv_gate.bias.data.fill_(1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        w = self.sigmoid(self.conv_gate(fusion))
        return w * x1 + (1 - w) * x2

class InnovativeMultiScaleCrossAttention(nn.Module):
    # <─ DEĞİŞİKLİK: Yeni 'cross_scale_init' parametresi eklendi.
    def __init__(self, dim, scales=(1, 2, 4), bias=False, downsample_factor=2, cross_scale_init=0.15):
        super(InnovativeMultiScaleCrossAttention, self).__init__()
        self.dim = dim
        self.scales = scales
        self.downsample_factor = downsample_factor
        self.local_attn_blocks = nn.ModuleDict()
        self.cross_attn_blocks = nn.ModuleDict()
        self.dynamic_fusion_blocks = nn.ModuleDict()
        self.se_blocks = nn.ModuleDict({str(s): SEBlock(dim) for s in self.scales})
        
        # <─ DEĞİŞİKLİK: Öğrenilebilir ölçeklendirme parametresi eklendi.
        self.cross_scale = nn.Parameter(torch.tensor(cross_scale_init))

        for s_val in scales:
            self.local_attn_blocks[str(s_val)] = LocalAttention(dim=dim, sample_rate=1, bias=bias)
            self.cross_attn_blocks[str(s_val)] = CrossAttention(dim=dim, bias=bias)
            self.dynamic_fusion_blocks[str(s_val)] = DynamicFusion(in_channels=dim)
        self.scale_fusion = nn.Conv2d(dim * len(scales), dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        outs = []
        for s_val in self.scales:
            effective_ds_factor = s_val * self.downsample_factor
            
            if effective_ds_factor > 1:
                actual_pool_factor = max(1, int(round(effective_ds_factor)))
                x_small = F.avg_pool2d(x, kernel_size=actual_pool_factor, stride=actual_pool_factor)
                local_feat_small = self.local_attn_blocks[str(s_val)](x_small)
                cross_feat_small = self.cross_attn_blocks[str(s_val)](x_small, x_small)
                
                if x_small.shape[2] != H or x_small.shape[3] != W:
                    local_feat_up = F.interpolate(local_feat_small, size=(H, W), mode='bilinear', align_corners=False)
                    cross_feat_up = F.interpolate(cross_feat_small, size=(H, W), mode='bilinear', align_corners=False)
                else:
                    local_feat_up = local_feat_small
                    cross_feat_up = cross_feat_small
                
                # <─ DEĞİŞİKLİK: Cross-attention özelliği füzyondan önce ölçeklendiriliyor.
                cross_feat_up = self.cross_scale * cross_feat_up
                fused = self.dynamic_fusion_blocks[str(s_val)](local_feat_up, cross_feat_up)
                fused = fused + local_feat_up
                fused = self.se_blocks[str(s_val)](fused)
                outs.append(fused)
            else:
                local_feat = self.local_attn_blocks[str(s_val)](x)
                cross_feat = self.cross_attn_blocks[str(s_val)](x, x)
                
                # <─ DEĞİŞİKLİK: Cross-attention özelliği füzyondan önce ölçeklendiriliyor.
                cross_feat = self.cross_scale * cross_feat
                fused = self.dynamic_fusion_blocks[str(s_val)](local_feat, cross_feat)
                fused = fused + local_feat
                fused = self.se_blocks[str(s_val)](fused)
                outs.append(fused)
                
        out_cat = torch.cat(outs, dim=1)
        out = self.scale_fusion(out_cat)
        return out

# ... (Kalan tüm sınıflar ve GridFormer mimarisi aynı kalıyor) ...

##############################################################################
## TransformerBlock Güncellemesi (LayerScale + DropPath ile)

class TransformerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor, 
                 bias, 
                 LayerNorm_type,
                 drop_path_prob=0.0,
                 init_values=1e-4,
                 sample_rate=2,
                 with_cp=True,
                 use_innovative_attn=False,
                 scales=(1, 2, 4),
                 downsample_factor=2):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        
        if use_innovative_attn:
            self.attn = InnovativeMultiScaleCrossAttention(dim=dim, scales=scales, bias=bias, downsample_factor=downsample_factor)
        else:
            self.attn = LocalAttention(dim=dim, sample_rate=sample_rate, bias=bias)
            
        self.drop1 = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.gamma1 = nn.Parameter(torch.ones(dim) * init_values) if init_values > 0 else None
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)
        self.drop2 = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.gamma2 = nn.Parameter(torch.ones(dim) * init_values) if init_values > 0 else None
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x_inner): 
            attn_out = self.attn(self.norm1(x_inner))
            if self.gamma1 is not None:
                x_res = x_inner + self.drop1(self.gamma1.view(1, -1, 1, 1) * attn_out)
            else:
                x_res = x_inner + self.drop1(attn_out)
            ffn_out = self.ffn(self.norm2(x_res))
            if self.gamma2 is not None:
                x_final = x_res + self.drop2(self.gamma2.view(1, -1, 1, 1) * ffn_out)
            else:
                x_final = x_res + self.drop2(ffn_out)
            return x_final
            
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)

##############################################################################
## DownSample ve UpSample

class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2,
                      kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2,
                      kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)

##############################################################################
## TBM (Transformer Block Module)

class TBM(nn.Module):
    def __init__(self, 
                 dim, 
                 num_blocks=2,
                 sample_rate=2, 
                 ffn_expansion_factor=2.66,
                 bias=False, 
                 use_innovative_attn=False,
                 scales=(1, 2, 4),
                 downsample_factor=2,
                 drop_path_prob=0.0,
                 layer_scale_init_values=1e-4):
        super(TBM, self).__init__()
        layers = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, num_blocks)] if num_blocks > 1 else [drop_path_prob]

        for i in range(num_blocks):
            current_block_drop_path_prob = dpr[i]
            blk = TransformerBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type='WithBias',
                sample_rate=sample_rate,
                with_cp=True,
                use_innovative_attn=use_innovative_attn,
                scales=scales,
                downsample_factor=downsample_factor,
                drop_path_prob=current_block_drop_path_prob,
                init_values=layer_scale_init_values
            )
            layers.append(blk)
        self.transformer = nn.Sequential(*layers)

    def forward(self, x):
        return self.transformer(x)

##############################################################################
## make_dense

class make_dense(nn.Module):
    def __init__(self, 
                 nChannels, 
                 growthRate=24,
                 num_blocks=4, 
                 ffn_expansion_factor=2.66,
                 bias=False, 
                 sample_rate=2,
                 kernel_size=1,
                 use_innovative_attn=False,
                 scales=(1, 2, 4),
                 downsample_factor=2,
                 drop_path_prob=0.0,
                 layer_scale_init_values=1e-4):
        super(make_dense, self).__init__()
        self.Transformer = TBM(
            dim=nChannels,
            num_blocks=num_blocks,
            sample_rate=sample_rate,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            use_innovative_attn=use_innovative_attn,
            scales=scales,
            downsample_factor=downsample_factor,
            drop_path_prob=drop_path_prob,
            layer_scale_init_values=layer_scale_init_values
        )
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        out = self.Transformer(x)
        out = self.conv(out)
        out = F.relu(out)
        out = torch.cat((x, out), 1)
        return out

##############################################################################
## RDTB (Residual Dense Transformer Block)

class RDTB(nn.Module):
    def __init__(self, 
                 nChannels, 
                 nDenselayer, 
                 num_blocks,
                 heads,
                 sample_rate,
                 growthRate,
                 use_innovative_attn=False,
                 scales=(1, 2, 4),
                 downsample_factor=2,
                 drop_path_prob=0.0,
                 layer_scale_init_values=1e-4):
        super(RDTB, self).__init__()
        nChannels_ = nChannels
        modules = []
        dpr_per_denselayer = [x.item() for x in torch.linspace(0, drop_path_prob, nDenselayer)] if nDenselayer > 1 else [drop_path_prob]

        for i in range(nDenselayer):
            modules.append(make_dense(nChannels=nChannels_,
                                      growthRate=growthRate,
                                      num_blocks=num_blocks,
                                      sample_rate=sample_rate,
                                      use_innovative_attn=use_innovative_attn,
                                      scales=scales,
                                      downsample_factor=downsample_factor,
                                      drop_path_prob=dpr_per_denselayer[i],
                                      layer_scale_init_values=layer_scale_init_values
                                      ))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        def _rdb_forward(x_inner):
            out = self.dense_layers(x_inner)
            out = self.conv_1x1(out)
            out = F.relu(out)
            return out + x_inner

        if self.training and x.requires_grad:
            return cp.checkpoint(_rdb_forward, x, use_reentrant=False)
        else:
            return _rdb_forward(x)

##############################################################################
## extractor / reconstruct

class extractor(nn.Module):
    def __init__(self, in_c, n_feat, num_blocks, sample_rate, heads=1,
                 use_innovative_attn=False, scales=(1,2,4), downsample_factor=2,
                 drop_path_prob=0.0, layer_scale_init_values=1e-4):
        super(extractor, self).__init__()
        self.conv_in = nn.Conv2d(in_c, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.tbm_in = TBM(dim=n_feat, num_blocks=num_blocks, sample_rate=sample_rate,
                           ffn_expansion_factor=2.66, bias=False, 
                           use_innovative_attn=use_innovative_attn,
                           scales=scales, downsample_factor=downsample_factor,
                           drop_path_prob=drop_path_prob,
                           layer_scale_init_values=layer_scale_init_values)
    def forward(self, x):
        x = self.conv_in(x)
        x = self.tbm_in(x)
        return x

class reconstruct(nn.Module):
    def __init__(self, in_c, out_c, num_blocks, sample_rate, heads=1, kernel_size=3,
                 use_innovative_attn=False, scales=(1,2,4), downsample_factor=2,
                 drop_path_prob=0.0, layer_scale_init_values=1e-4):
        super(reconstruct, self).__init__()
        self.tbm_out = TBM(dim=in_c, num_blocks=num_blocks, sample_rate=sample_rate,
                           ffn_expansion_factor=2.66, bias=False, 
                           use_innovative_attn=use_innovative_attn,
                           scales=scales, downsample_factor=downsample_factor,
                           drop_path_prob=drop_path_prob,
                           layer_scale_init_values=layer_scale_init_values)
        self.conv_out = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        x = self.tbm_out(x)
        x = self.conv_out(x)
        return x

##############################################################################
## GridFormer Ana Mimari
@ARCH_REGISTRY.register()
class GridFormer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=64,
                 kernel_size=3,
                 stride=2,
                 height=3,
                 width=6,
                 num_blocks=[1,2,4],
                 growthRate=24,
                 nDenselayer=3,
                 heads=[2,4,8],
                 attention=True,
                 sample_rate_trunk=[4,2,2], 
                 use_innovative_attn=True,
                 scales=(1,2,4),
                 downsample_factor=2,
                 drop_path_prob=0.1,
                 layer_scale_init_values=1e-4,
                 **kwargs):
        super(GridFormer, self).__init__()
        
        if kwargs:
            print(f"GridFormer Info: Received and ignoring unexpected keyword arguments: {kwargs}")

        self.height = height
        self.width  = width
        self.stride = stride 
        self.dim    = dim    
        self.use_innovative_attn = use_innovative_attn
        self.scales = scales 
        self.downsample_factor = downsample_factor 

        self.TBM_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.extractor_module = nn.ModuleDict()
        self.reconstruct_module = nn.ModuleDict()

        max_channels_in_grid = dim * (stride ** (height - 1))
        self.coefficient = nn.Parameter(
            torch.ones(height, width, 2, max_channels_in_grid),
            requires_grad=attention
        )

        self.conv_in = nn.Conv2d(in_channels, dim, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.conv_out = nn.Conv2d(dim, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

        self.TBM_in = RDTB(nChannels=dim, nDenselayer=nDenselayer, num_blocks=num_blocks[0],
                            growthRate=growthRate, heads=heads[0], sample_rate=sample_rate_trunk[0],
                            use_innovative_attn=use_innovative_attn, scales=scales, 
                            downsample_factor=downsample_factor,
                            drop_path_prob=drop_path_prob, 
                            layer_scale_init_values=layer_scale_init_values)
        self.TBM_out = RDTB(nChannels=dim, nDenselayer=nDenselayer, num_blocks=num_blocks[0],
                             growthRate=growthRate, heads=heads[0], sample_rate=sample_rate_trunk[0],
                             use_innovative_attn=use_innovative_attn, scales=scales, 
                             downsample_factor=downsample_factor,
                             drop_path_prob=drop_path_prob, 
                             layer_scale_init_values=layer_scale_init_values)

        current_TBM_level_channels = dim
        for i in range(height):
            for j in range(width - 1):
                self.TBM_module[f'{i}_{j}'] = RDTB(
                    nChannels=current_TBM_level_channels,
                    nDenselayer=nDenselayer,
                    num_blocks=num_blocks[i],
                    growthRate=growthRate,
                    heads=heads[i],          
                    sample_rate=sample_rate_trunk[i], 
                    use_innovative_attn=use_innovative_attn,
                    scales=scales,
                    downsample_factor=downsample_factor,
                    drop_path_prob=drop_path_prob, 
                    layer_scale_init_values=layer_scale_init_values
                )
            if i < height - 1:
                 current_TBM_level_channels *= stride

        _in_channels_for_down_path = dim
        for i in range(height - 1):
            next_grid_level_idx = i + 1
            n_feat_for_extractor = dim * (stride ** next_grid_level_idx)
            self.downsample_module[f'{i}_0'] = DownSample(_in_channels_for_down_path) 
            self.extractor_module[f'{i}_0'] = extractor(
                in_c=in_channels,
                n_feat=n_feat_for_extractor,
                num_blocks=num_blocks[next_grid_level_idx],
                sample_rate=sample_rate_trunk[next_grid_level_idx],
                use_innovative_attn=use_innovative_attn,
                scales=scales,
                downsample_factor=downsample_factor,
                drop_path_prob=drop_path_prob,
                layer_scale_init_values=layer_scale_init_values
            )
            for j_ds in range(1, width // 2):
                 self.downsample_module[f'{i}_{j_ds}'] = DownSample(_in_channels_for_down_path)
            _in_channels_for_down_path *= stride

        _in_channels_for_up_path = dim * (stride**(height-1))
        for i in range(height - 2, -1, -1):
            source_grid_level_for_reconstruct_input = i + 1 
            in_c_for_reconstruct = dim * (stride ** source_grid_level_for_reconstruct_input)
            self.upsample_module[f'{i}_{width-1}'] = UpSample(_in_channels_for_up_path) 
            self.reconstruct_module[f'{i}_{width-1}'] = reconstruct(
                in_c=in_c_for_reconstruct,
                out_c=out_channels,
                num_blocks=num_blocks[source_grid_level_for_reconstruct_input],
                sample_rate=sample_rate_trunk[source_grid_level_for_reconstruct_input],
                kernel_size=kernel_size,
                use_innovative_attn=use_innovative_attn,
                scales=scales,
                downsample_factor=downsample_factor,
                drop_path_prob=drop_path_prob,
                layer_scale_init_values=layer_scale_init_values
            )
            for j_us in range(width // 2, width - 1): 
                 self.upsample_module[f'{i}_{j_us}'] = UpSample(_in_channels_for_up_path)
            _in_channels_for_up_path //= stride

    def forward_features(self, x):
        Image_index = [None for _ in range(self.height)]
        Image_index[0] = x
        for i in range(1, self.height):
            Image_index[i] = F.interpolate(Image_index[i-1], scale_factor=0.5, mode='bilinear', align_corners=False)
        inp = self.conv_in(x)
        x_index = [[None for _ in range(self.width)] for _ in range(self.height)]
        x_index[0][0] = self.TBM_in(inp)
        for j in range(1, self.width // 2):
            x_index[0][j] = self.TBM_module[f'0_{j-1}'](x_index[0][j-1])
        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module[f'{i-1}_0'](x_index[i-1][0]) + \
                            self.extractor_module[f'{i-1}_0'](Image_index[i])
            for j in range(1, self.width // 2):
                term1_horizontal = self.TBM_module[f'{i}_{j-1}'](x_index[i][j-1]) 
                term2_vertical = self.downsample_module[f'{i-1}_{j}'](x_index[i-1][j]) 
                channel_num_current = term1_horizontal.shape[1]
                coef0 = self.coefficient[i, j, 0, :channel_num_current][None, :, None, None]
                coef1 = self.coefficient[i, j, 1, :channel_num_current][None, :, None, None]
                x_index[i][j] = coef0 * term1_horizontal + coef1 * term2_vertical
        mid_col_idx_left_half_end = self.width // 2 - 1
        x_index[self.height-1][mid_col_idx_left_half_end+1] = \
            self.TBM_module[f'{self.height-1}_{mid_col_idx_left_half_end}'](x_index[self.height-1][mid_col_idx_left_half_end])
        for j in range(mid_col_idx_left_half_end + 2, self.width):
            x_index[self.height-1][j] = self.TBM_module[f'{self.height-1}_{j-1}'](x_index[self.height-1][j-1])
        for i in range(self.height - 2, -1, -1):
            term1_horizontal_edge = self.TBM_module[f'{i}_{mid_col_idx_left_half_end}'](x_index[i][mid_col_idx_left_half_end]) 
            term2_vertical_edge = self.upsample_module[f'{i}_{mid_col_idx_left_half_end+1}'](x_index[i+1][mid_col_idx_left_half_end+1]) 
            channel_num_current_edge = term1_horizontal_edge.shape[1]
            coef0_edge = self.coefficient[i, mid_col_idx_left_half_end+1, 0, :channel_num_current_edge][None, :, None, None]
            coef1_edge = self.coefficient[i, mid_col_idx_left_half_end+1, 1, :channel_num_current_edge][None, :, None, None]
            x_index[i][mid_col_idx_left_half_end+1] = coef0_edge * term1_horizontal_edge + coef1_edge * term2_vertical_edge
            for j in range(mid_col_idx_left_half_end + 2, self.width):
                term1_horizontal = self.TBM_module[f'{i}_{j-1}'](x_index[i][j-1]) 
                term2_vertical = self.upsample_module[f'{i}_{j}'](x_index[i+1][j]) 
                channel_num_current = term1_horizontal.shape[1]
                coef0 = self.coefficient[i, j, 0, :channel_num_current][None, :, None, None]
                coef1 = self.coefficient[i, j, 1, :channel_num_current][None, :, None, None]
                x_index[i][j] = coef0 * term1_horizontal + coef1 * term2_vertical
        
        image_out = [None for _ in range(self.height)]
        
        final_feature_map_lvl0 = self.TBM_out(x_index[0][self.width-1])
        image_out[0] = self.conv_out(final_feature_map_lvl0) + Image_index[0] 
        
        for i_out_lvl in range(1, self.height): 
            reconstruct_module_key = f'{i_out_lvl-1}_{self.width-1}'
            image_out[i_out_lvl] = self.reconstruct_module[reconstruct_module_key](x_index[i_out_lvl][self.width-1]) + Image_index[i_out_lvl]
            
        return image_out

    def forward(self, x):
        outs = self.forward_features(x)
        outs.reverse()   
        return outs