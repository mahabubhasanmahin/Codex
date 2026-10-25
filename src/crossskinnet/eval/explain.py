"""Grad-CAM image explanations and field-coalition SHAP metadata attributions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn

from crossskinnet.data.metadata import FIELDS, SLICES
from crossskinnet.io import write_csv
from crossskinnet.models.network import CrossSkinNet
from crossskinnet.train.dataset import SkinDataset


class FixedMetadata(nn.Module):
    """Adapt a multimodal classifier to the image-only Grad-CAM callable contract."""

    def __init__(self, model: CrossSkinNet, meta: torch.Tensor, mask: torch.Tensor) -> None:
        super().__init__()
        self.model, self.meta, self.mask = model, meta, mask

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image, self.meta.expand(len(image), -1), self.mask.expand(len(image), -1))['logits']


def _reshape(model: CrossSkinNet, tensor: torch.Tensor) -> torch.Tensor:
    """Convert Swin/ViT activations into spatial maps for pytorch-grad-cam."""
    encoder = model.image_encoder
    if encoder.format == 'nhwc':
        return tensor.permute(0, 3, 1, 2)
    if encoder.format == 'tokens':
        registers = int(getattr(encoder.model.config, 'num_register_tokens', 0))
        tokens = tensor[:, 1 + registers:]
        size = math.isqrt(tokens.shape[1])
        if size * size != tokens.shape[1]:
            raise ValueError('Grad-CAM requires a square spatial patch-token grid')
        return tokens.transpose(1, 2).reshape(len(tensor), tensor.shape[-1], size, size)
    return tensor


def _shap_fields(model: CrossSkinNet, item: dict[str, Any], target: int,
                 device: torch.device, samples: int, seed: int) -> tuple[np.ndarray, float, float]:
    """Explain observed field availability for a fixed image and predicted class.

    Background is the same image with all metadata absent, not fabricated patient
    data or test-set averages. Coalitions remove entire semantic fields. The
    resulting attributions include the gate transition to the image-only path.
    """
    import shap

    meta = item['metadata'].numpy()
    observed = item['mask'].numpy()
    image = item['image'].unsqueeze(0).to(device)

    def predict(coalitions: np.ndarray) -> np.ndarray:
        result = []
        with torch.no_grad():
            for start in range(0, len(coalitions), 32):
                masks = coalitions[start:start + 32] * observed
                values = np.repeat(meta[None], len(masks), axis=0)
                for index, region in enumerate(SLICES):
                    values[:, region] *= masks[:, index, None]
                logits = model(image.expand(len(masks), -1, -1, -1),
                    torch.tensor(values, device=device, dtype=torch.float32),
                    torch.tensor(masks, device=device, dtype=torch.float32))['logits']
                result.extend(logits.softmax(-1)[:, target].cpu().tolist())
        return np.asarray(result)

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        explainer = shap.KernelExplainer(predict, np.zeros((1, len(FIELDS))))
        values = np.asarray(explainer.shap_values(np.ones((1, len(FIELDS))), nsamples=samples,
                                                l1_reg=0, silent=True)).reshape(-1)
    finally:
        np.random.set_state(state)
    baseline = float(np.asarray(explainer.expected_value).item())
    prediction = float(predict(np.ones((1, len(FIELDS))))[0])
    return values, baseline, prediction


def explain(model: CrossSkinNet, dataset: SkinDataset, probability: np.ndarray,
            output: Path, options: dict[str, Any], seed: int, device: torch.device) -> None:
    """Save up to N correct/N wrong Grad-CAMs and metadata SHAP plots plus CSVs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    output.mkdir(parents=True, exist_ok=True)
    count = int(options.get('explanations_per_outcome', 10))
    if count < 1:
        raise ValueError('explanations_per_outcome must be positive when explainability is enabled')
    rng = np.random.default_rng(seed)
    actual = np.array([dataset.classes.index(row['unified_label']) for row in dataset.rows])
    predicted = probability.argmax(1)
    chosen = []
    selection = []
    for outcome, candidates in [('correct', np.flatnonzero(actual == predicted)), ('wrong', np.flatnonzero(actual != predicted))]:
        indices = rng.choice(candidates, min(count, len(candidates)), replace=False)
        print(f'Explainability {outcome}: requested {count}, available {len(candidates)}, selected {len(indices)}')
        chosen.extend((int(index), outcome) for index in indices)
        selection.append(dict(outcome=outcome, available=len(candidates), selected=len(indices)))
    write_csv(output / 'selection_counts.csv', selection)
    records = []
    means = np.zeros(len(FIELDS))
    model.eval()
    for index, outcome in chosen:
        item = dataset[index]
        row = dataset.rows[index]
        image = item['image'].unsqueeze(0).to(device).requires_grad_(True)
        wrapper = FixedMetadata(model, item['metadata'][None].to(device), item['mask'][None].to(device))
        with GradCAM(model=wrapper, target_layers=[model.image_encoder.cam_layer()],
                     reshape_transform=lambda value: _reshape(model, value)) as cam:
            heatmap = cam(input_tensor=image, targets=[ClassifierOutputTarget(int(predicted[index]))])[0]
        pixels = item['image'].permute(1, 2, 0).numpy()
        mean = np.array(dataset.cfg['training'].get('image_mean', [.485, .456, .406]))
        std = np.array(dataset.cfg['training'].get('image_std', [.229, .224, .225]))
        pixels = np.clip(pixels * std + mean, 0, 1).astype(np.float32)
        Image.fromarray(show_cam_on_image(pixels, heatmap, use_rgb=True)).save(output / f'{index}_{outcome}_gradcam.png')
        values, baseline, prediction = _shap_fields(model, item, int(predicted[index]), device,
            int(options.get('shap_samples', 512)), seed + index)
        means += np.abs(values)
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.barh(FIELDS, values)
        axis.set_xlabel('SHAP contribution to predicted-class probability')
        figure.tight_layout()
        figure.savefig(output / f'{index}_{outcome}_shap.png')
        plt.close(figure)
        for field, value in zip(FIELDS, values):
            records.append(dict(image_id=row['image_id'], source_dataset=row['source_dataset'], outcome=outcome,
                predicted_label=dataset.classes[predicted[index]], field=field, shap_value=float(value),
                baseline_probability=baseline, predicted_probability=prediction,
                reconstruction_error=float(abs(baseline + values.sum() - prediction))))
    write_csv(output / 'metadata_shap.csv', records, ['image_id', 'source_dataset', 'outcome', 'predicted_label',
        'field', 'shap_value', 'baseline_probability', 'predicted_probability', 'reconstruction_error'])
    summary = [dict(field=field, mean_absolute_shap=float(value / max(1, len(chosen)))) for field, value in zip(FIELDS, means)]
    write_csv(output / 'metadata_shap_global.csv', summary)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.barh(FIELDS, [r['mean_absolute_shap'] for r in summary])
    axis.set_title('Selected correct/wrong examples: mean absolute field SHAP')
    figure.tight_layout()
    figure.savefig(output / 'metadata_shap_global.png')
    plt.close(figure)
