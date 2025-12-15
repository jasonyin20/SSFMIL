import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from mamba_ssm import Mamba
from einops import rearrange, repeat, einsum
from modules.four_path_mamba_v2.mamba.mamba_ssm import SRMamba
from modules.four_path_mamba_v2.mamba.mamba_ssm.modules.four_path_mamba import FPMamba
from modules.four_path_mamba_v2.mamba.mamba_ssm.modules.bimamba import BiMamba
from modules.four_path_mamba_v2.mamba.mamba_ssm.modules.mamba_simple import Mamba
from modules.four_path_mamba_v2.torch_wavelets import IDWT_2D, DWT_2D


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim  # 768
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        #self.sparsity_threshold = 0.01
        self.sparsity_threshold = 0.1


        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)
        #print(x.shape)
        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')  # FFT on N dimension
        #print(x.shape,"fft2")

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0])

        #print(x_real_1.shape)
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1])
        #print(x_imag_1.shape)


        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[1]) + \
                   self.complex_bias_2[0]

        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[0]) + \
                   self.complex_bias_2[1]

        #print(x_real_2.shape)
        #print(x_imag_2.shape)

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()


        #x = torch.stack([x_real_1, x_imag_1], dim=-1).float()

        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x

        #print(x.shape)
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        #print(data.shape)
        return x



class WaveletFilter(nn.Module):
    def __init__(self, dim, h=16, w=16 ,c=4):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.c = c
        self.comples_weight = nn.Parameter(torch.randn(self.c, h, w,  2, dtype=torch.float32) * 0.02)
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')


    def _set(self, h, w):
        self.h = h
        self.w = w
        self.comples_weight = nn.Parameter(torch.randn(h, w, self.c, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, spatial_size=None):
        B, C ,N = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(C, B, a, b)
        x = x.to(torch.float32)
        # print(x.shape, '1wx')          #[600, 1, 32, 32]
        x = self.dwt(x)
        #print(x.shape, '2wx')          #[600, 4, 16, 16]
        # print(self.comples_weight.shape, 'self.comples_weight')




        weight = torch.view_as_complex(self.comples_weight).cuda()
        x = x * weight.float()
        # print(x.shape, '3wx')                                 #[600, 4, 16, 16]
        x = self.idwt(x)
        # print(x.shape, '4wx')   #[600, 1, 32, 32]
        x = x.reshape(B, C, N)

        return x




class Block_mamba(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)


        # self.mlp2 = nn.Sequential(*[
        #     nn.LayerNorm(dim),
        #     WaveletFilter(dim=dim)
        # ])

        # self.attn  = VisionMamba_v4_path(
        #     embed_dim=dim, depth=1, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        #     final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        #     if_cls_token=False, if_devide_out=False, use_middle_cls_token=False,if_bidirectional=False)

        self.attn  = FPMamba(
           d_model=dim,
           d_state=16,
           d_conv=4,
           expand=2,
        )

        # self.attn1 = FPMamba(
        #     d_model=dim,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2,
        # )



        self.mlp = EinFFT(dim)
        #self.mlp1 = EinFFT(dim)

        #self.mlp = FFN(dim,dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape)

        #t_x , weights = self.attn(x)
        x = x + self.drop_path(self.attn(x))

        x = x + self.drop_path(self.mlp(self.norm2(x)))


        # x = x + self.drop_path(self.attn1(x))
        # #
        # x = x + self.drop_path(self.mlp1(self.norm2(x)))
        #



        #print(x.shape)
        return x

class Block_mamba_with_fusion(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = FPMamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        fusion_gate_type = "sigmoid"
        mode = "full"

        self.mlp = EinFFT(dim)
        self.mode = mode
        self.fusion_gate_type = fusion_gate_type
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)


        if mode == 'full' and fusion_gate_type != 'none':
            if fusion_gate_type == 'learned':
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            elif fusion_gate_type == 'attention':
                self.fusion_attn = nn.Sequential(
                    nn.Linear(dim * 2, dim // 4),
                    nn.ReLU(),
                    nn.Linear(dim // 4, 2),
                    nn.Softmax(dim=-1)
                )
            elif fusion_gate_type == 'sigmoid':
                self.fusion_gate = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.Sigmoid()
                )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _apply_fusion_gate(self, x_mse, x_aff):
        if self.fusion_gate_type == 'none':
            return x_mse + x_aff
        elif self.fusion_gate_type == 'learned':
            weight = torch.sigmoid(self.fusion_weight)
            return weight * x_mse + (1 - weight) * x_aff
        elif self.fusion_gate_type == 'attention':
            combined = torch.cat([x_mse, x_aff], dim=-1)
            weights = self.fusion_attn(combined)
            return weights[:, :, 0:1] * x_mse + weights[:, :, 1:2] * x_aff
        elif self.fusion_gate_type == 'sigmoid':
            gate = self.fusion_gate(x_mse + x_aff)
            return gate * x_mse + (1 - gate) * x_aff
        else:
            return x_mse + x_aff

    def forward(self, x):
        # print(x.shape)

        identity = x

        x_MSE = self.drop_path(self.attn(x))

        x_AFF = self.drop_path(self.mlp(self.norm2(x)))

        if self.fusion_gate_type == 'none':
            combined = x_MSE + x_AFF
            x = identity + combined
        else:
            x_fused = self._apply_fusion_gate(x_MSE, x_AFF)
            x = identity + x_fused

        return x








class Block_bi_mamba(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)



        #bimamba
        self.attn = BiMamba(
             d_model=dim,
             d_state=16,
             d_conv=4,
             expand=2,
        )





        #self.mlp = EinFFT(dim)
        #self.mlp = FFN(dim,dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape)

        #t_x , weights = self.attn(x)
        x = x + self.drop_path(self.attn(x))


        #print(x.shape)
        return x


class Block_valia_mamba(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)




        #mamba
        self.attn  = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )


        # mamba
        # self.attn  = SRMamba(
        #     d_model=dim,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2,
        # )


        #self.mlp = EinFFT(dim)
        #self.mlp = FFN(dim,dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape)

        #t_x , weights = self.attn(x)
        x = x + self.drop_path(self.attn(x))

        #x = x + self.drop_path(self.mlp(self.norm2(x)))

        #print(x.shape)
        return x


class Block_sr_mamba(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #self.norm2 = norm_layer(dim)


        # srmamba
        self.attn  = SRMamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        #self.mlp = EinFFT(dim)
        # self.mlp = FFN(dim,dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape)

        x = x + self.drop_path(self.attn(x))

        #x = x + self.drop_path(self.mlp(self.norm2(x)))

        # print(x.shape)
        return x

class Block_four_mamba(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #self.norm2 = norm_layer(dim)


        # self.mlp2 = nn.Sequential(*[
        #     nn.LayerNorm(dim),
        #     WaveletFilter(dim=dim)
        # ])

        # self.attn  = VisionMamba_v4_path(
        #     embed_dim=dim, depth=1, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        #     final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        #     if_cls_token=False, if_devide_out=False, use_middle_cls_token=False,if_bidirectional=False)

        self.attn  = FPMamba(
           d_model=dim,
           d_state=16,
           d_conv=4,
           expand=2,
        )

        # self.attn1 = FPMamba(
        #     d_model=dim,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2,
        # )



        #self.mlp = EinFFT(dim)
        #self.mlp1 = EinFFT(dim)

        #self.mlp = FFN(dim,dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape)

        #t_x , weights = self.attn(x)
        x = x + self.drop_path(self.attn(x))

        #x = x + self.drop_path(self.mlp(self.norm2(x)))


        #x = x + self.drop_path(self.attn1(x))

        #x = x + self.drop_path(self.mlp1(self.norm2(x)))




        #print(x.shape)
        return x
if __name__ == '__main__':
    bag_label = torch.tensor(1)
    data = torch.randn((1, 6000, 1024)).cuda()
    # # ---->pad
    # H = data.shape[1]
    # _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    # add_length = _H * _W - H
    # h = torch.cat([data, data[:, :add_length, :]], dim=1)  # [B, N, 512]
    model = Block_mamba(dim=1024,drop_path=0.2).cuda()
    data =model(data)

