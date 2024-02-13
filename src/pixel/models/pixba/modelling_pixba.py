"""
PyTorch PIXEL models
"""

import collections
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
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

from pixel.utils import DependencyParsingModelOutput, format_mask

from pixel.models import Biaffine
from pixel.models import PoolingForSequenceClassificationHead, PoolingMode
from pixel.models import ViTModel
from pixel.models import PIXELConfig

from mamba_ssm import Mamba

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


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.
    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


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
        return x


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

    def random_masking(self, sequence, attention_mask):
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
        noise_mask = torch.argmax(noise * attention_mask, dim=1)
        noise[torch.arange(noise.size(0)), noise_mask] = 100.0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore

    def controlled_masking(self, sequence, attention_mask, patch_mask):

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
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore

    def forward(self, pixel_values, attention_mask=None, patch_mask=None):
        batch_size, num_channels, height, width = pixel_values.shape
        print(f'pixel dimensions before embedding:{pixel_values.shape}')
        embeddings = self.patch_embeddings(pixel_values)
        print(f'pixel dimensions after embedding:{embeddings.shape}')

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        if patch_mask is not None:
            embeddings, attention_mask, mask, ids_restore = self.controlled_masking(
                embeddings, attention_mask, patch_mask
            )
        else:
            embeddings, attention_mask, mask, ids_restore = self.random_masking(embeddings, attention_mask)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        return embeddings, attention_mask, mask, ids_restore


# MAMBA BLOCK
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

class PIXBABlock(nn.Module):

    def __init__(self,config, d_state=16, d_conv=4, expand=2):
        super(PIXBABlock,self).__init__()

        self.d_model = config.d_model
        self.mamba = Mamba(
                self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,

                )
    
    def forward(self,x):

        y = self.mamba(x)

        return y



class PIXBAEncoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_state=16, d_conv=4, expand=2):
        super(PIXBAEncoder, self).__init__()
        self.layers = nn.ModuleList([PIXBABlock(d_model, d_state, d_conv, expand) for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


