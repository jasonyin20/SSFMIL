# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
from torch import Tensor

from einops import rearrange, repeat

from modules.four_path_mamba_v2.mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from einops import rearrange


class Direct_Mamba_v1(nn.Module):
    def __init__(
            self,
            d_model,
            n_directions=1,
            bidirectional=True,
            d_state=16,
            d_conv=4,
            expand=4,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            conv_init=None,

    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.conv_init = conv_init

        self.nheads = 2
        self.n_ssms = 4


        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)


        # ----> Build conv1d for each direction


        # ----> Build SSM block for each direction
        self.n_directions = n_directions
        self.bidirectional = bidirectional
        self.n_ssms = self.n_directions * (2 if bidirectional else 1)
        #print(self.n_ssms)

        self.conv1d_list = nn.ModuleList()
        for _ in range(self.n_ssms):
            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(conv1d.weight, -self.conv_init, self.conv_init)
            self.conv1d_list.append(conv1d)

        self.activation = "silu"
        self.act = nn.SiLU()



        self.x_proj_list = nn.ModuleList()
        for _ in range(self.n_ssms):
            x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.x_proj_list.append(x_proj)


        self.dt_proj_list = nn.ModuleList()
        for _ in range(self.n_ssms):
            dt_proj = nn.Linear(
                self.dt_rank, self.d_inner , bias=True, **factory_kwargs
            )
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError
            self.dt_proj_list.append(dt_proj)





        # ----> Build dt_bias for each direction
        # Initialize log dt bias
        self.dt_bias_list = nn.ParameterList()
        for _ in range(self.n_ssms):
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            dt_bias = nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            dt_bias._no_weight_decay = True
            self.dt_bias_list.append(dt_bias)



        # ----> Build A_log for each direction
        A_init_range=(1, 16)

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        self.A_log_list = nn.ParameterList()
        for _ in range(self.n_ssms):
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
            self.A_log_list.append(A_log)


        # ----> Build D for each direction
        self.D_list = nn.ParameterList()
        for _ in range(self.n_ssms):
            # D "skip" parameter
            D = nn.Parameter(torch.ones(self.d_inner, device=device))
            D._no_weight_decay = True
            self.D_list.append(D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, order_index_list=None, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):

        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        assert len(order_index_list) == self.n_directions if order_index_list is not None else True

        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = hidden_states.shape
        else:
            batch_seqlen, dim = hidden_states.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None

        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # zxbcdt = self.in_proj(hidden_states)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        # if seqlen_og is not None:
        #     zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        #

        zxbcdt = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        #print(zxbcdt.shape)

        if self.in_proj.bias is not None:
            zxbcdt = zxbcdt + rearrange(self.in_proj.bias.to(dtype=zxbcdt.dtype), "d -> d 1")
        # If the model is loaded in fp16, without the .float() here, A might be -inf



        A_list = [-torch.exp(A_log.float()) for A_log in self.A_log_list] # (nheads) or (d_inner, d_state)

        # A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # A_b = -torch.exp(self.A_b_log.float())
        # A_c = -torch.exp(self.A_c_log.float())
        # A_dd = -torch.exp(self.A_dd_log.float())

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states

            out_list = []
            for i in range(self.n_ssms):
                if self.bidirectional:
                    if order_index_list is not None:
                        order_index = order_index_list[i // 2]
                    else:
                        order_index = torch.arange(seqlen, device=zxbcdt.device)
                    if i % 2 == 1:
                        order_index = order_index.flip(0)
                else:
                    if order_index_list is not None:
                        order_index = order_index_list[i]
                    else:
                        order_index = torch.arange(seqlen, device=zxbcdt.device)
                #print(zxbcdt.shape)

                zxbcdt_i = zxbcdt[:,:,order_index]  # B, L, C
                conv1d_weight = self.conv1d_list[i].weight
                conv1d_bias = self.conv1d_list[i].bias
                x_proj_weight = self.x_proj_list[i].weight
                dt_proj_weight = self.dt_proj_list[i].weight

                dt_bias = self.dt_bias_list[i]

                A = A_list[i]
                D = self.D_list[i]

                # print(zxbcdt_i.shape)
                # print(conv1d_weight.shape)
                # print(conv1d_bias.shape)
                # print(x_proj_weight.shape)
                # print(dt_proj_weight.shape)
                # print(A.shape)
                # print(D.shape)
                # print(dt_bias.shape)


                out = mamba_inner_fn_no_out_proj(
                    zxbcdt_i,
                    conv1d_weight,
                    conv1d_bias,
                    x_proj_weight,
                    dt_proj_weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    D.float(),
                    delta_bias=dt_bias.float(),
                    delta_softplus=True,
                )

                reverse_order_index = torch.argsort(order_index)
                out = out[:, :,reverse_order_index]
                out_list.append(out)

            out = torch.mean(torch.stack(out_list, dim=0), dim=0)
            #print(out.shape)
            # print(f'out.shape: {out.shape}')
            #out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
            out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            #print(out.shape)







        else:
            # x, z拆分用于两个分支的聚合
            x, z = hidden_states.chunk(2, dim=1)

            x_b = x.flip([-1])
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            # 为无causal_conv1d_fn提供了手动的处理方案
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
                x = self.act(self.conv1d_b(x_b)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                x_b = causal_conv1d_fn(
                    x=x_b,
                    weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    bias=self.conv1d_b.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            # SSM内核操作，分别为前向和反向生成对应的B和C
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            y_b = selective_scan_fn(
                x_b,
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            y_b = rearrange(y_b, "b d l -> b l d")
            out = self.out_proj(y_b + y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

if __name__ == '__main__':
    model =  Direct_Mamba_v1(
                n_directions=4,
                bidirectional=True,
                d_model=512,
                d_state=16,
                d_conv=4,
                expand=2).cuda()
    data = torch.randn(1,100,512).cuda()
    result = model(data)