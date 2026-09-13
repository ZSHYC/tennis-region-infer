"""默认网球事件模型的纯推理网络。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(F.gelu(self.conv(values))) + residual
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class _Heads(nn.Module):
    """仅用于集中构造与发布权重同名、同形状的两只输出头。"""

    def _init_heads(self) -> None:
        self.eventness_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, 1)
        )
        self.type_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, 2)
        )

    def _heads(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "eventness_logit": self.eventness_head(pooled).squeeze(-1),
            "type_logits": self.type_head(pooled),
        }


class EventModel(_Heads):
    """11 维轨迹双头模型；保留发布权重中 779 维投影的完整参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(779, 64)
        self.input_norm = nn.LayerNorm(64)
        self.input_dropout = nn.Dropout(0.25)
        self.convs = nn.ModuleList(_ResidualConv() for _ in range(3))
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self._init_heads()

    def forward(self, trajectory: torch.Tensor) -> dict[str, torch.Tensor]:
        if trajectory.ndim != 3 or trajectory.shape[-1] != 11:
            raise ValueError("trajectory 必须是 [B,T,11]")
        valid = trajectory[..., 9] > 0.5
        values = F.linear(trajectory, self.proj.weight[:, :11], self.proj.bias)
        values = self.input_dropout(F.gelu(self.input_norm(values))).transpose(1, 2)
        for block in self.convs:
            values = block(values)
        values, _ = self.gru(values.transpose(1, 2))
        mask = valid[..., None]
        temporal_mean = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        temporal_max = values.masked_fill(~mask, -torch.inf).amax(dim=1)
        temporal_max = torch.where(
            torch.isfinite(temporal_max), temporal_max, torch.zeros_like(temporal_max)
        )
        center = values[:, values.shape[1] // 2]
        return self._heads(torch.cat((center, temporal_mean, temporal_max), dim=-1))


class VisualTemporalModel(_Heads):
    """五视图 DINOv3 特征的区域优先时序双头模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(768, 64)
        self.input_norm = nn.LayerNorm(64)
        self.input_dropout = nn.Dropout(0.25)
        self.position_proj = nn.Linear(2, 64, bias=False)
        self.register_buffer(
            "tile_positions",
            torch.tensor(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))),
            persistent=False,
        )
        self.convs = nn.ModuleList(_ResidualConv() for _ in range(3))
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self._init_heads()

    def forward(self, visual: torch.Tensor) -> dict[str, torch.Tensor]:
        if visual.ndim != 3 or visual.shape[-1] != 3840:
            raise ValueError("visual 必须是 [B,T,3840]")
        tokens = visual.reshape(*visual.shape[:2], 5, 768).to(self.proj.weight.dtype)
        projected = self.input_norm(self.proj(tokens))
        batch_size, steps, regions, hidden = projected.shape
        values = projected.permute(0, 2, 3, 1).reshape(batch_size * regions, hidden, steps)
        for block in self.convs:
            values = block(values)
        projected = values.reshape(batch_size, regions, hidden, steps).permute(0, 3, 1, 2)
        global_token = projected[:, :, 0]
        keys = projected[:, :, 1:] + self.position_proj(self.tile_positions)
        scores = (global_token.unsqueeze(2) * keys).sum(dim=-1) / math.sqrt(hidden)
        weights = scores.softmax(dim=-1)
        values = global_token + (weights.unsqueeze(-1) * keys).sum(dim=2)
        values = self.input_dropout(F.gelu(self.input_norm(values))).transpose(1, 2)
        values, _ = self.gru(values.transpose(1, 2))
        pooled = torch.cat(
            (values[:, values.shape[1] // 2], values.mean(dim=1), values.amax(dim=1)),
            dim=-1,
        )
        return self._heads(pooled)
