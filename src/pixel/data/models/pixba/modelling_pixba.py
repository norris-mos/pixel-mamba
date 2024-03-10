"""
PyTorch PIXEL models
"""

import collections
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange, repeat

import numpy as np
import torch
import wandb

from utils.misc import get_2d_sincos_pos_embed

from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import ViTForImageClassification, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

# from pixel.utils import DependencyParsingModelOutput, format_mask

# from biaffine import Biaffine
# from pooling import PoolingForSequenceClassificationHead, PoolingMode
# from vit import ViTModel
#from ..pixel.configuration_pixel import PIXELConfig

# from mamba_ssm import Mamba
from pixba.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from pixba.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from pixba.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


logger = logging.getLogger(__name__)


#EMBEDDING CLASSES  
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return x, x





def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PIXBAPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    Based on timm implementation, which can be found here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x # should be - (B, 529, 768)


class PIXBAEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PIXBAPatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches ** 0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # Attention mask indicates patches containing actual text
        # Out of the patches containing actual text we take the one with the highest noise
        # And bump up noise to 100 to guarantee that it gets masked
        # We therefore ensure that at least one masked patch has actual text
        # This is necessary because we only compute loss on patches having text, i.e. loss would otherwise be NaN
        print(noise.shape)
        
        # Harsh - since we are not using masking I don't think we need to increase the noise.
        # noise_mask = torch.argmax(noise * attention_mask, dim=1)
        # noise[torch.arange(noise.size(0)), noise_mask] = 100.0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        # attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # return sequence_masked, attention_mask_masked, mask, ids_restore
        return sequence_masked, mask, ids_restore

    def controlled_masking(self, sequence, patch_mask):

        batch_size, seq_length, dim = sequence.shape

        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        # We keep the interface the same as in the original random_masking function above
        # The only difference is that instead of random noise we use the predefined mask
        # Sometimes the greedy span masking yields fewer masked patches than specified through mask_ratio
        # We additionally mask out the difference between them randomly using noise in [0, 0.01]
        noise = patch_mask + (torch.rand(batch_size, seq_length, device=sequence.device) / 100)  # noise in [0, 0.01)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        #attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, mask, ids_restore

    def forward(
            self,
            pixel_values,
            # attention_mask=None,
            patch_mask=None
            ):
        batch_size, num_channels, height, width = pixel_values.shape
        print(f'pixel dimensions before embedding:{pixel_values.shape}')
        embeddings = self.patch_embeddings(pixel_values)
        print(f'pixel dimensions after embedding:{embeddings.shape}')

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        if patch_mask is not None:
            #embeddings, attention_mask, mask, ids_restore = self.controlled_masking(
            embeddings, mask, ids_restore = self.controlled_masking(
                embeddings,
                # attention_mask,
                patch_mask
            )
        else:
            embeddings, mask, ids_restore = self.random_masking(embeddings) #, attention_mask)
            #embeddings, attention_mask, mask, ids_restore = self.random_masking(embeddings, attention_mask)
        print("Embeddings after masking - ", embeddings.shape)
        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        #attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        #return embeddings, attention_mask, mask, ids_restore
        print("PIXELEmbeddings return embeddings of dimension - ", embeddings.shape)
        return embeddings, mask, ids_restore # embeddings - (B, 529, 768)


# MAMBA BLOCK
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# class PIXBABlock(nn.Module):

#     def __init__(self,config, d_state=16, d_conv=4, expand=2):
#         super(PIXBABlock,self).__init__()

#         self.d_model = config.d_model
#         self.mamba = Mamba(
#                 self.d_model,
#                 d_state=d_state,
#                 d_conv=d_conv,
#                 expand=expand,

#                 )
    
#     def forward(self,x):

#         y = self.mamba(x)

#         return y
    

class PIXBABlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
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

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        #print("executing forward")
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            #print("using fast path")
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            #print("executing selective_scan")
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
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    
class PIXBABlockWrapper(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class PIXBAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([ #each layer returns (out: out_proj)
            PIXBABlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dt_rank=config.dt_rank,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
                dt_init=config.dt_init,
                dt_scale=config.dt_scale,
                dt_init_floor=config.dt_init_floor,
                conv_bias=config.conv_bias,
                bias=config.bias,
                use_fast_path=config.use_fast_path,
                layer_idx=i,
                device=config.device,
                dtype=config.dtype,
            )
            for i in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, src, inference_params=None):
        # src should be embeddings_output: (B, 529, 768)
        print("Input to encoder - ",src.shape)
        for layer in self.layers: # In Mamba paper, hidden_states from previous layers are normalised before going into the next layer - ref: mamba_simple @line 349. Even in PIXEL is see similar thing happening in modelling_pixel.js @line841 - Harsh
            src = layer(src, inference_params=inference_params) # src is now - out_proj (B, 397, 768)
        src = self.norm(src)
        return src


class PIXBADecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder_embed = nn.Linear(config.hidden_size, config.d_model, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.d_model), requires_grad=False
        )  # fixed sin-cos embedding

        self.layers = nn.ModuleList([
            PIXBABlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dt_rank=config.dt_rank,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
                dt_init=config.dt_init,
                dt_scale=config.dt_scale,
                dt_init_floor=config.dt_init_floor,
                conv_bias=config.conv_bias,
                bias=config.bias,
                use_fast_path=config.use_fast_path,
                layer_idx=i,
                device=config.device,
                dtype=config.dtype,
            )
            for i in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Identity()
        self.initialize_weights(config.num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches ** 0.5), add_cls_token=True
        )
        x = torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        print('unsqueezing output of sin/cos - ', x.shape)
        self.decoder_pos_embed.data.copy_(x)
        print("decoder position embedding - ", self.decoder_pos_embed.shape)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(self, encoder_output, ids_restore, return_token_num=0, inference_params=None):
        x = self.decoder_embed(encoder_output)
        print("After inserting decoder embedding - ", x.shape)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        print("after adding mask token - ", x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        print("after restoring ids/patch - ", x_.shape)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        print("Hidden state after inserting mask_tokens/hidden patches - ", x.shape)
        print("pos embedding - ", self.decoder_pos_embed.shape)
        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        src = hidden_states
        print("Input to decoder - ", src.shape)
        for layer in self.layers:
            src = layer(src, inference_params=inference_params)
        src = self.norm(src)
        #src = self.head(src[:, -return_token_num:])
        src = self.head(src)

        # remove cls token
        src = src[:, 1:, :]

        print("Out of decoder - ", src.shape)
        return src

"""

class PIXBAEncoder(nn.Module):
    def __init__(self, config, num_blocks):
        super(PIXBAEncoder, self).__init__()
        d_model = config.d_model
        d_state = config.d_state
        d_conv = config.d_conv
        expand = config.expand
        self.layers = nn.ModuleList([PIXBABlock(d_model, d_state, d_conv, expand) for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
"""
"""
#More complex architecture for encoding... however does not align as well with original PIXEL encoder structure
class PIXBAEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 3,
        norm: str = "layer",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len

        self.inp = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList([
            PIXBABlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm=norm,
            )
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self._emb_dim = d_model

    def init_hidden_state(self, batch_size: int, device: torch.device):
        conv_states, ssm_states = [], []
        for block in self.blocks:
            conv_state, ssm_state = block.allocate_inference_cache(
                batch_size, max_seqlen=self.max_seq_len
            )
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        return PIXBAHiddenState(conv_states, ssm_states)

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, PIXBAHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(self, seq, time_idxs=None, hidden_state=None):
        seq = self.inp(seq)
        if hidden_state is None:
            for block in self.blocks:
                seq = block(seq)
        else:
            assert not self.training
            assert isinstance(hidden_state, PIXBAHiddenState)
            for i, block in enumerate(self.blocks):
                conv_state_i, ssm_state_i = hidden_state[i]
                seq, new_conv_state_i, new_ssm_state_i = block.step(
                    seq, conv_state=conv_state_i, ssm_state=ssm_state_i
                )
                hidden_state[i] = new_conv_state_i, new_ssm_state_i
        return self.out_norm(seq), hidden_state


class PIXBAHiddenState:
    def __init__(self, conv_states, ssm_states):
        self.conv_states = conv_states
        self.ssm_states = ssm_states

    def reset(self, idxs):
        for conv_state, ssm_state in zip(self.conv_states, self.ssm_states):
            conv_state[idxs] = 0
            ssm_state[idxs] = 0
        """

#TODO FIX MODEL to corresopond to Enocoder & Decoder
class PIXBAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = PIXBAEmbeddings(config)
        self.encoder = PIXBAEncoder(config)

    def forward(
        self,
        pixel_values = None,
        # attention_mask = None,
        head_mask = None,
        patch_mask = None,
        #output_attentions = None,
        #output_hidden_states = None,
        return_dict = None
    ):
        embedding_output, mask, ids_restore = self.embeddings(pixel_values, patch_mask)
        encoder_outputs = self.encoder(
            embedding_output,
            #output_hidden_states = output_hidden_states,
            #return_dict = return_dict,
        )
        print("Encoder return output of dimension - ", encoder_outputs.shape)
        sequence_output = encoder_outputs
        #sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]
        
        return PIXBAModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
        )
    
@dataclass
class PIXBAForPreTrainingOutput(ModelOutput):
    """
    Class for PIXELForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are attended to and which are not.
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    # attention_mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class PIXBAModelOutput(ModelOutput):
    """
    Class for PIXELModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
                        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
                    `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    #attentions: Optional[Tuple[torch.FloatTensor]] = None

class PIXBAForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vit = PIXBAModel(config)
        self.decoder = PIXBADecoder(config)#, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values=None,
        # attention_mask=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            # attention_mask=attention_mask,
            head_mask=head_mask,
            patch_mask=patch_mask,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)#, attention_mask)  # [N, L, p*p*3]
        logits = decoder_outputs #decoder_outputs.logits

        #merged_mask = torch.bitwise_and(mask == 1, attention_mask == 1).long()
        #loss = self.forward_loss(pixel_values, logits, merged_mask)
        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PIXBAForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            #attention_mask=attention_mask,
            ids_restore=ids_restore,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )

