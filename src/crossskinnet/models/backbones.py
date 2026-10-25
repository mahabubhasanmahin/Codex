"""Pooled and token embeddings from tiny, timm and Hugging Face image encoders."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """A common spatial-token interface; tiny never accesses the network."""

    def __init__(self, name: str, options: dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        self.frozen = bool(options.get('frozen', False))
        self.format = 'nchw'
        if name == 'tiny':
            self.model = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                                       nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((4, 4)))
            self.dimension = 32
        elif name in ('efficientnet', 'swin'):
            import timm
            if not options.get('id'):
                raise ValueError(f'Configure model.backbones.{name}.id')
            self.model = timm.create_model(options['id'], pretrained=options.get('pretrained', True), num_classes=0)
            self.dimension = self.model.num_features
            self.format = 'nhwc' if name == 'swin' else 'nchw'
        elif name in ('dinov2', 'dinov3'):
            from transformers import AutoModel
            if not options.get('id'):
                raise ValueError(f'Configure model.backbones.{name}.id')
            try:
                self.model = AutoModel.from_pretrained(options['id'], token=os.environ.get('HF_TOKEN'),
                    local_files_only=options.get('local_files_only', False))
            except OSError as error:
                raise ValueError(f'Cannot load {name} {options["id"]}. For gated DINOv3, accept the model license '
                                 'and set HF_TOKEN securely; offline runs need cached weights.') from error
            self.dimension = self.model.config.hidden_size
            self.format = 'tokens'
        else:
            raise ValueError(f'Unsupported backbone {name}')
        if self.frozen:
            self.model.requires_grad_(False)

    def train(self, mode: bool = True) -> ImageEncoder:
        """Keep frozen batch statistics and dropout fixed during head training."""
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a pooled embedding and spatial image tokens."""
        if self.format == 'tokens':
            features = self.model(pixel_values=image).last_hidden_state
            registers = int(getattr(self.model.config, 'num_register_tokens', 0))
            return features[:, 0], features[:, 1 + registers:]
        features = self.model(image) if self.name == 'tiny' else self.model.forward_features(image)
        if features.ndim != 4:
            raise ValueError(f'{self.name}: expected 4-D spatial features, got {features.shape}')
        tokens = features.flatten(2).transpose(1, 2) if self.format == 'nchw' else features.flatten(1, 2)
        return tokens.mean(1), tokens

    def cam_layer(self) -> nn.Module:
        """Locate a spatial layer before final pooling/CLS aggregation for Grad-CAM."""
        if self.name == 'tiny':
            return self.model[2]
        if self.name == 'efficientnet':
            return self.model.conv_head
        if self.name == 'swin':
            return self.model.layers[-1].blocks[-1].norm1
        # Hugging Face ViTs expose the encoder under either encoder or a model prefix.
        base = getattr(self.model, 'encoder', None)
        if base is None:
            base = getattr(self.model, 'model', None)
        if base is None and hasattr(self.model, 'layer'):
            base = self.model
        if base is None:
            raise ValueError(f'{self.name}: configure/adapt Grad-CAM target for this transformers architecture')
        block = base.layer[-1]
        return block.norm1 if hasattr(block, 'norm1') else block.layernorm_before
