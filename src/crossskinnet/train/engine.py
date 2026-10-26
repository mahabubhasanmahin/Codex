"""Deterministic training, resumable epoch checkpoints and validation-only calibration."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from crossskinnet.io import config, read_csv, report, write_csv
from crossskinnet.models.network import CrossSkinNet
from crossskinnet.train.dataset import SkinDataset
from crossskinnet.data.sources import validate_source_rows, validate_training_coverage


def validate_integrity(root: Path) -> None:
    """Require a review-cleared dataset produced by the current build stage."""
    manifest = root / 'dataset_integrity.json'
    if not manifest.exists():
        raise ValueError('Missing dataset_integrity.json; rerun review-aware stages 04 and 05')
    record = json.loads(manifest.read_text())
    if record.get('policy') != 'review' or not record.get('ready'):
        raise ValueError('Dataset is not cleared for duplicate review')
    for name in ('metadata.csv','zero_shot.csv'):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != record.get('files', {}).get(name):
            raise ValueError(f'{name} differs from the cleared dataset; rebuild stage 05')


def seed_all(seed: int) -> None:
    """Set all relevant generators and deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class FocalLoss(nn.Module):
    """Class-weighted focal loss with unweighted probability modulation."""

    def __init__(self, weights: torch.Tensor, gamma: float = 2) -> None:
        super().__init__()
        self.register_buffer('weights', weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probability = torch.log_softmax(logits.float(), -1).gather(1, labels[:, None]).squeeze(1)
        return (-self.weights[labels] * (1 - log_probability.exp()).pow(self.gamma) * log_probability).mean()


def make_model(cfg: dict[str, Any], backbone: str, classes: list[str], frozen: bool | None = None) -> CrossSkinNet:
    """Build the exact architecture from persisted configuration."""
    options = dict(cfg['model']['backbones'][backbone])
    if frozen is not None:
        options['frozen'] = frozen
    return CrossSkinNet(len(classes), backbone, options, int(cfg['model']['dimension']), int(cfg['model']['heads']))


def batch_forward(model: CrossSkinNet, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move just model inputs to the selected device."""
    return model(*(batch[key].to(device) for key in ('image', 'metadata', 'mask')))


def calibrate(model: CrossSkinNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Fit image-only temperature with validation labels only; never test labels."""
    if loader.dataset.split != 'val':
        raise ValueError('Calibration requires validation split only')
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits.append(batch_forward(model, batch, device)['image_logits'].detach())
            labels.append(batch['label'].to(device))
    values, targets = torch.cat(logits), torch.cat(labels)
    loss = nn.CrossEntropyLoss()
    before = float(loss(values, targets))
    log_temperature = nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=40)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        result = loss(values / log_temperature.exp().clamp(.05, 20), targets)
        result.backward()
        return result

    optimizer.step(closure)
    temperature = log_temperature.exp().clamp(.05, 20).detach()
    after = float(loss(values / temperature, targets))
    if after > before:
        temperature = torch.ones_like(temperature)
        after = before
    model.temperature.copy_(temperature)
    return {'temperature': float(temperature), 'validation_nll_before': before, 'validation_nll_after': after}


def run(config_path: Path, backbone: str = 'tiny', epochs: int | None = None,
        resume: Path | None = None, frozen: bool | None = None) -> Path:
    """Train, save every completed epoch, restore best validation F1, then calibrate."""
    cfg = config(config_path)
    if epochs is not None:
        cfg['training']['epochs'] = epochs
    seed_all(int(cfg['seed']))
    torch.set_num_threads(int(cfg['training'].get('cpu_threads', 2)))
    device = torch.device(cfg['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    root = Path(cfg['paths'].get('unified_root') or cfg['output'] / 'CrossSkinNet_Unified_Dataset')
    active_status = cfg['output'] / 'dedup_status.json'
    if active_status.exists() and not json.loads(active_status.read_text()).get('ready'):
        raise ValueError('Duplicate review incomplete in output_root; do not train a provisional or stale dataset')
    validate_integrity(root)
    rows = read_csv(root / 'metadata.csv')
    validate_source_rows(rows, cfg, 'train')
    validate_training_coverage([row for row in rows if row['split']=='train'], cfg)
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid development split {row['split']}")
        group_splits.setdefault(row['group_id'], set()).add(row['split'])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError('Patient/lesion group leakage in metadata.csv')
    classes = sorted({r['unified_label'] for r in rows if r['split'] == 'train'})
    if len(classes) < 2:
        raise ValueError('Training requires at least two classes')
    model = make_model(cfg, backbone, classes, frozen).to(device)
    run_dir = cfg['output'] / 'runs' / f"{backbone}_{'frozen' if model.image_encoder.frozen else 'finetuned'}"
    if (run_dir / 'last.pt').exists() and resume is None:
        raise ValueError(f'{run_dir} already contains checkpoints; use --resume or a different output_root')
    run_dir.mkdir(parents=True, exist_ok=True)
    options = cfg['training']
    train = SkinDataset(root, classes, 'train', cfg, training=True)
    validation = SkinDataset(root, classes, 'val', cfg)
    if not len(train) or not len(validation):
        raise ValueError('Both train and validation splits must be nonempty')
    generator = torch.Generator().manual_seed(int(cfg['seed']))
    loader = DataLoader(train, batch_size=int(options['batch_size']), shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(validation, batch_size=int(options['batch_size']), num_workers=0)
    backbone_params = [p for n, p in model.named_parameters() if n.startswith('image_encoder.') and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith('image_encoder.') and p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': options['lr'] * options['backbone_lr_factor']},
                                  {'params': head_params, 'lr': options['lr']}], weight_decay=options['weight_decay'])
    total_epochs = int(options['epochs'])
    if total_epochs < 1:
        raise ValueError('epochs must be positive')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    amp = device.type == 'cuda' and bool(options.get('mixed_precision', True))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    counts = np.bincount([classes.index(r['unified_label']) for r in train.rows], minlength=len(classes))
    weights = torch.tensor(len(train) / (len(classes) * counts), device=device, dtype=torch.float32)
    criterion = FocalLoss(weights, float(options.get('focal_gamma', 2)))
    signature = hashlib.sha256((root / 'metadata.csv').read_bytes()).hexdigest()
    start_epoch, best, patience = 0, -1.0, 0
    history = []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint['signature'] != signature or checkpoint['classes'] != classes or checkpoint['backbone'] != backbone:
            raise ValueError('Resume checkpoint dataset/classes/backbone do not match')
        old_options = dict(checkpoint['config']['training'])
        if old_options != options or checkpoint['frozen'] != model.image_encoder.frozen:
            raise ValueError('Resume requires identical training hyperparameters, epoch schedule and frozen mode')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best, patience = checkpoint['epoch'] + 1, checkpoint['best_f1'], checkpoint['patience']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint['cuda_rng']])
        random.setstate(checkpoint['random_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        train.rng.bit_generator.state = checkpoint['metadata_rng']
        generator.set_state(checkpoint['loader_rng'].cpu())
        history = checkpoint['history']
        # Preserve a best checkpoint when resuming into a restored run directory.
        if not (run_dir / 'best.pt').exists():
            raise ValueError('Resume requires the run directory with best.pt and last.pt intact')
    started = time.monotonic()
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                result = batch_forward(model, batch, device)
                targets = batch['label'].to(device)
                loss = criterion(result['logits'], targets)
                # Auxiliary image supervision trains fallback even when no sample drops all fields.
                loss = loss + float(options.get('image_aux_weight', .25)) * criterion(result['image_logits'], targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(targets)
        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = batch_forward(model, batch, device)['logits']
                actual.extend(batch['label'].tolist())
                predicted.extend(logits.argmax(-1).cpu().tolist())
        score = float(f1_score(actual, predicted, labels=list(range(len(classes))), average='macro', zero_division=0))
        scheduler.step()
        improved = score > best
        best = max(best, score)
        patience = 0 if improved else patience + 1
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(train), val_macro_f1=score,
                            learning_rate=optimizer.param_groups[-1]['lr']))
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), epoch=epoch, best_f1=best, patience=patience, classes=classes, config=cfg,
            backbone=backbone, frozen=model.image_encoder.frozen, signature=signature, history=history,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            random_rng=random.getstate(), numpy_rng=np.random.get_state(), metadata_rng=train.rng.bit_generator.state,
            loader_rng=generator.get_state())
        torch.save(checkpoint, run_dir / f'epoch_{epoch + 1:03}.pt')
        torch.save(checkpoint, run_dir / 'last.pt')
        if improved:
            torch.save(checkpoint, run_dir / 'best.pt')
        write_csv(run_dir / 'metrics.csv', history)
        print(history[-1], flush=True)
        if patience >= int(options['patience']) or time.monotonic() - started > float(options.get('max_hours', 11)) * 3600:
            break
    best_checkpoint = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model'])
    calibration = calibrate(model, val_loader, device)
    best_checkpoint['model'] = model.state_dict()
    best_checkpoint['calibration'] = calibration
    torch.save(best_checkpoint, run_dir / 'best_calibrated.pt')
    write_csv(run_dir / 'calibration.csv', [calibration])
    (run_dir / 'classes.json').write_text(json.dumps(classes))
    report('Epoch metrics', history)
    print('Calibration (validation only):', calibration)
    return run_dir / 'best_calibrated.pt'
