"""In-domain, zero-shot, missingness, explainability and backbone result artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from crossskinnet.data.metadata import MODES
from crossskinnet.eval.metrics import bootstrap_f1, classification_metrics
from crossskinnet.io import config, read_csv, report, write_csv
from crossskinnet.train.dataset import SkinDataset
from crossskinnet.train.engine import batch_forward, make_model, seed_all, validate_integrity


def predict(model: torch.nn.Module, dataset: SkinDataset, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Infer without updating weights or calibration on the evaluation data."""
    probability, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            probability.extend(batch_forward(model, batch, device)['logits'].softmax(-1).cpu().numpy())
            targets.extend(batch['label'].tolist())
    return np.asarray(targets, dtype=int), np.asarray(probability).reshape(-1, len(dataset.classes))


def _confusion_plot(matrix: list[list[int]], classes: list[str], path: Path) -> None:
    """Render a labeled confusion matrix without assuming every class is observed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(max(5, len(classes) * .5), max(4, len(classes) * .5)))
    axis.imshow(matrix, cmap='Blues')
    axis.set_xticks(range(len(classes)), classes, rotation=90)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('Actual')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run(config_path: Path, checkpoint_path: Path, explainability: bool = True) -> Path:
    """Evaluate a trusted local checkpoint against its original split and vocab."""
    cfg = config(config_path)
    seed_all(int(cfg['seed']))
    torch.set_num_threads(int(cfg['training'].get('cpu_threads', 2)))
    options = cfg.get('evaluation', {})
    device = torch.device(cfg['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    root = Path(cfg['paths'].get('unified_root') or cfg['output'] / 'CrossSkinNet_Unified_Dataset')
    validate_integrity(root)
    signature = hashlib.sha256((root / 'metadata.csv').read_bytes()).hexdigest()
    if signature != checkpoint['signature']:
        raise ValueError('Evaluation metadata differs from the training split; use the original dataset')
    # Preprocessing and architecture are taken from training, never silently changed.
    inference_cfg = dict(checkpoint['config'])
    inference_cfg['evaluation'] = options
    model = make_model(inference_cfg, checkpoint['backbone'], classes, checkpoint['frozen']).to(device)
    model.load_state_dict(checkpoint['model'])
    tag = f"{checkpoint['backbone']}_{'frozen' if checkpoint['frozen'] else 'finetuned'}"
    destination = cfg['output'] / 'results' / tag
    destination.mkdir(parents=True, exist_ok=True)
    detailed = {}
    summaries, robustness, comparison = [], [], []
    exclusions = []
    in_domain = None
    for split in ('test', 'zero_shot'):
        baseline = None
        for mode in MODES:
            dataset = SkinDataset(root, classes, split, inference_cfg, mode=mode)
            targets, probability = predict(model, dataset, device, int(cfg['training']['batch_size']))
            metric = classification_metrics(targets, probability, classes)
            key = f'{split}_{mode}'
            detailed[key] = metric
            scalar = {k: v for k, v in metric.items() if k not in ('per_class_f1', 'confusion_matrix')}
            summaries.append(dict(split=split, ablation=mode, **scalar))
            if mode == 'full':
                baseline = metric['macro_f1']
                if split == 'test':
                    in_domain = (dataset, probability)
                excluded = read_csv(root / ('zero_shot.csv' if split == 'zero_shot' else 'metadata.csv'))
                excluded = [r for r in excluded if r['split'] == split and r['unified_label'] not in classes]
                exclusions.extend(dict(source_dataset=r['source_dataset'], image_id=r['image_id'],
                    unified_label=r['unified_label'], reason='no_training_counterpart') for r in excluded)
                lo, hi = bootstrap_f1(targets, probability, [r['group_id'] for r in dataset.rows],
                    int(cfg['seed']), int(options.get('bootstrap_repetitions', 1000)))
                comparison.append(dict(backbone=checkpoint['backbone'], frozen=checkpoint['frozen'], split=split,
                    macro_f1=metric['macro_f1'], ci95_low=lo, ci95_high=hi, samples=len(targets),
                    excluded_samples=dataset.excluded, dataset_signature=signature))
                if len(targets):
                    _confusion_plot(metric['confusion_matrix'], classes, destination / f'{split}_confusion.png')
            drop = None if baseline is None or metric['macro_f1'] is None else baseline - metric['macro_f1']
            robustness.append(dict(split=split, ablation=mode, macro_f1=metric['macro_f1'], drop_vs_observed_full=drop))
            predictions = []
            for index, row in enumerate(dataset.rows):
                predictions.append(dict(image_id=row['image_id'], source_dataset=row['source_dataset'],
                    group_id=row['group_id'], actual=row['unified_label'], predicted=classes[probability[index].argmax()],
                    **{f'probability_{label}':float(value) for label, value in zip(classes, probability[index])}))
            write_csv(destination / f'{key}_predictions.csv', predictions,
                ['image_id','source_dataset','group_id','actual','predicted',*[f'probability_{label}' for label in classes]])
            for source in sorted({r['source_dataset'] for r in dataset.rows}):
                indices = np.array([i for i, row in enumerate(dataset.rows) if row['source_dataset'] == source])
                source_metric = classification_metrics(targets[indices], probability[indices], classes)
                detailed[f'{key}_{source}'] = source_metric
                summaries.append(dict(split=split, ablation=mode, source_dataset=source,
                    **{k:v for k,v in source_metric.items() if k not in ('per_class_f1','confusion_matrix')}))
    (destination / 'metrics.json').write_text(json.dumps(detailed, indent=2, allow_nan=False))
    write_csv(destination / 'metrics.csv', summaries)
    write_csv(destination / 'robustness.csv', robustness)
    write_csv(destination / 'zero_shot_exclusions.csv', exclusions,
              ['source_dataset','image_id','unified_label','reason'])
    per_class = [dict(evaluation=key, unified_label=label, f1=value) for key, metric in detailed.items()
                 for label, value in metric['per_class_f1'].items()]
    write_csv(destination / 'per_class_f1.csv', per_class, ['evaluation','unified_label','f1'])
    comparison_path = cfg['output'] / 'results/backbone_comparison.csv'
    old = read_csv(comparison_path) if comparison_path.exists() else []
    retained = [r for r in old if not (r['backbone'] == checkpoint['backbone'] and
                r['frozen'].lower() == str(checkpoint['frozen']).lower() and r['dataset_signature'] == signature)]
    write_csv(comparison_path, retained + comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 4))
    for split in ('test','zero_shot'):
        values = [r['drop_vs_observed_full'] for r in robustness if r['split'] == split]
        axis.plot(MODES, [np.nan if value is None else value for value in values], marker='o', label=split)
    axis.tick_params(axis='x', rotation=45)
    axis.set_ylabel('Macro-F1 drop from observed full metadata')
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / 'robustness.png')
    plt.close(figure)
    if explainability and in_domain is not None and len(in_domain[0]):
        from crossskinnet.eval.explain import explain
        explain(model, *in_domain, destination / 'explanations', options, int(cfg['seed']), device)
    (destination / 'evaluation_manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint_path),
        dataset_signature=signature, calibration=checkpoint.get('calibration'), explainability=explainability,
        baseline='Observed available metadata; structurally missing fields remain missing.',
        bootstrap='Patient/lesion group resampling; fixed class vocabulary.',
        shap='Field-presence coalitions against all-missing fallback, conditioned on each image.'),indent=2))
    report('Evaluation summary', summaries)
    print('Zero-shot excluded:', len(exclusions))
    return destination
