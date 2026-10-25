"""CrossSkinNet: learned missing tokens, gated cross-attention and image fallback."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from crossskinnet.data.metadata import SLICES, WIDTHS
from crossskinnet.models.backbones import ImageEncoder


class MetadataEncoder(nn.Module):
    """One learned missing token and a small observed-value MLP per field."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(nn.Sequential(nn.Linear(width, dimension), nn.ReLU(),
            nn.Linear(dimension, dimension)) for width in WIDTHS)
        self.missing = nn.Parameter(torch.randn(len(WIDTHS), dimension) * .02)
        self.fields = nn.Parameter(torch.randn(len(WIDTHS), dimension) * .02)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace missing representations with learned tokens, never input averages."""
        observed = torch.stack([encoder(values[:, region]) for encoder, region in zip(self.encoders, SLICES)], 1)
        return torch.where(mask.bool().unsqueeze(-1), observed, self.missing.unsqueeze(0)) + self.fields.unsqueeze(0)


class CrossSkinNet(nn.Module):
    """Route each sample independently to full fusion or calibrated image-only logits."""

    def __init__(self, classes: int, backbone: str, options: dict[str, Any], dimension: int = 64, heads: int = 4) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError('Fusion dimension must be divisible by attention heads')
        self.image_encoder = ImageEncoder(backbone, options)
        self.image_projection = nn.Linear(self.image_encoder.dimension, dimension)
        self.metadata_encoder = MetadataEncoder(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.gate = nn.Linear(dimension * 2, dimension)
        self.fusion_norm = nn.LayerNorm(dimension)
        self.fusion_head = nn.Linear(dimension, classes)
        self.image_head = nn.Linear(dimension, classes)
        self.register_buffer('temperature', torch.ones(()))

    def forward(self, image: torch.Tensor, metadata: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits and image logits; probabilities are computed only for inference."""
        pooled, tokens = self.image_encoder(image)
        pooled = self.image_projection(pooled)
        tokens = self.image_projection(tokens)
        clinical = self.metadata_encoder(metadata, mask)
        attended, weights = self.attention(tokens, clinical, clinical)
        context = attended.mean(1)
        gate = torch.sigmoid(self.gate(torch.cat([pooled, context], -1)))
        fused = self.fusion_norm(pooled + gate * context)
        image_logits = self.image_head(pooled)
        available = mask.bool().any(1, keepdim=True)
        logits = torch.where(available, self.fusion_head(fused), image_logits / self.temperature.clamp_min(.05))
        return {'logits': logits, 'image_logits': image_logits, 'attention': weights}
