# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement in DINOv3-LICENSE.md.

"""发布模型所用的 DINOv3 ViT-B/16 纯推理骨干。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _flat_apply(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
    """与官方单元素列表路径的逐 token 计算保持一致。"""
    shape = values.shape
    return module(values.flatten(0, -2)).reshape(*shape[:-1], -1)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _Rope(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        periods = 100.0 ** (2 * torch.arange(16, dtype=torch.float32) / 32)
        self.register_buffer("periods", periods)

    def forward(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        options = {"device": self.periods.device, "dtype": torch.float32}
        rows = torch.arange(0.5, height, **options) / height
        columns = torch.arange(0.5, width, **options) / width
        coords = torch.stack(torch.meshgrid(rows, columns, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class _MaskedQkv(nn.Linear):
    def __init__(self) -> None:
        super().__init__(768, 2304, bias=True)
        mask = torch.ones_like(self.bias)
        mask[768:1536] = 0
        self.register_buffer("bias_mask", mask)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias * self.bias_mask.to(self.bias.dtype))


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = _MaskedQkv()
        self.proj = nn.Linear(768, 768)

    @staticmethod
    def _apply_rope(
        values: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        original_dtype = values.dtype
        sin, cos = rope
        values = values.to(sin.dtype)
        prefix = values.shape[-2] - sin.shape[-2]
        rotated = values[..., prefix:, :] * cos + _rotate_half(values[..., prefix:, :]) * sin
        return torch.cat((values[..., :prefix, :], rotated), dim=-2).to(original_dtype)

    def forward(
        self, values: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch, tokens, _ = values.shape
        qkv = self.qkv(values.flatten(0, -2)).reshape(batch, tokens, 3, 12, 64)
        query, key, value = (part.transpose(1, 2) for part in torch.unbind(qkv, 2))
        query = self._apply_rope(query, rope)
        key = self._apply_rope(key, rope)
        values = F.scaled_dot_product_attention(query, key, value)
        values = values.transpose(1, 2).reshape(batch, tokens, 768)
        return _flat_apply(self.proj, values)


class _LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((768,), 1e-5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.gamma


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.0)
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.drop(self.act(self.fc1(values)))
        return self.drop(self.fc2(values))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-5)
        self.attn = _Attention()
        self.ls1 = _LayerScale()
        self.norm2 = nn.LayerNorm(768, eps=1e-5)
        self.mlp = _Mlp()
        self.ls2 = _LayerScale()

    def forward(
        self, values: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        values = values + self.ls1(self.attn(_flat_apply(self.norm1, values), rope))
        return values + self.ls2(_flat_apply(self.mlp, _flat_apply(self.norm2, values)))


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


class DinoV3(nn.Module):
    """固定 ViT-B/16 结构，返回归一化的 CLS 特征。"""

    def __init__(self) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, 768))
        self.storage_tokens = nn.Parameter(torch.empty(1, 4, 768))
        self.mask_token = nn.Parameter(torch.empty(1, 768))
        self.patch_embed = _PatchEmbed()
        self.rope_embed = _Rope()
        self.blocks = nn.ModuleList(_Block() for _ in range(12))
        self.norm = nn.LayerNorm(768, eps=1e-5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 256, 256):
            raise ValueError("images 必须是 [N,3,256,256]")
        patches = self.patch_embed(images)
        batch = images.shape[0]
        cls = self.cls_token + 0 * self.mask_token
        values = torch.cat(
            (cls.expand(batch, -1, -1), self.storage_tokens.expand(batch, -1, -1), patches),
            dim=1,
        )
        for block in self.blocks:
            values = block(values, self.rope_embed(16, 16))
        return _flat_apply(self.norm, values)[:, 0]
