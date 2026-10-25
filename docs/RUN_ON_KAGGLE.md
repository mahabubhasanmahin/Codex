# Run CrossSkinNet on Kaggle

Start with the synthetic notebook, `notebooks/CrossSkinNet_Kaggle.ipynb`. It needs
no medical dataset or mapping decisions. Its random-data metrics are not thesis
results. The commands below assume your shell/notebook is in the repository root.

## 0. Get the implementation and dependencies

This task creates local Git commits; it does not push them to GitHub. Use the
provided implementation ZIP, or push these commits before cloning `main` below.
Uploading the ZIP as a private Kaggle Dataset and extracting it into writable
working storage is an alternative to cloning. Use its actual attached path.

```python
!git clone https://github.com/mahabubhasanmahin/Codex.git
%cd Codex
!python -m pip install -r requirements.txt
```

Kaggle normally already has PyTorch/torchvision. Keep those versions compatible.
Torch is for training, timm/transformers for real backbones, ImageHash for pHash,
and SHAP/grad-cam for the planned explanations. Installation can use Internet;
tests and the tiny demo do not download weights.

```python
!pytest -q
!python scripts/00_smoke.py --config configs/default.yaml
```

Look for `SYNTHETIC DEMO ONLY. Complete artifacts: ...` in the output. Open that
folder's `outputs/results/tiny_finetuned/` for plots, metrics and explanations.
Every demo uses a new directory under configured output_root; real mappings stay
unchanged. You do not need a GPU for this short demo.

## 1. Merge the real datasets

Attach all five sources at the configured paths. Enable a single GPU for later
training. Inspect `configs/default.yaml` and adjust only to actual mount paths.

```python
!python scripts/01_merge.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

## 2. Inventory and review labels, then apply

```python
!python scripts/02_labels.py inventory --config configs/default.yaml
```

Review `configs/label_mapping.csv`. Supply unified_label and coarse_group for
included entries, or set include=False with a reason. Unknown/ambiguous medical
mappings fail explicitly. Inventory preserves your edits.

```python
!python scripts/02_labels.py apply --config configs/default.yaml
```

## 3. Inventory and review sites, then normalize metadata

```python
!python scripts/03_metadata.py inventory --config configs/default.yaml
```

Map each null raw site in `configs/site_mapping.yaml` to an allowed anatomical
group, then run:

```python
!python scripts/03_metadata.py apply --config configs/default.yaml
```

## 4. Remove duplicate training images

```python
!python scripts/04_dedup.py --config configs/default.yaml
```

Inspect duplicates_report.csv, removed_training_duplicates.csv and duplicate_pairs.png.
ISIC/HAM overlap may remove most/all ISIC training images; report that honestly.
External pools are never pruned to retain training copies.

## 5. Build the unified dataset

```python
!python scripts/05_build_unified.py --config configs/default.yaml
```

Inspect dataset_stats.json and class-distribution plots in
`<output_root>/CrossSkinNet_Unified_Dataset/`. See upload_to_kaggle.md for private
hosting and paths.unified_root when reusing the derived dataset.

## 6. Train

First check tiny model execution; it is not a scientific backbone comparison.

```python
!python scripts/06_train.py --config configs/default.yaml --backbone tiny --epochs 1
```

Real backbone examples (separate runs; retain the identical dataset and schedule):

```python
!python scripts/06_train.py --config configs/default.yaml --backbone efficientnet --frozen
!python scripts/06_train.py --config configs/default.yaml --backbone efficientnet --no-frozen
```

Repeat for `swin`, `dinov2`, `dinov3`. Accept DINOv3's model license and supply
HF_TOKEN through Kaggle secrets/environment, never in a committed file. IDs are
configured in YAML. Pretrained downloads require Internet or a supported local
model cache/path. Prefer one backbone/mode per 12-hour session and retain its
entire run directory before ending the session; each epoch is checkpointed.

Resume uses the same config/backbone/frozen flags, adding the actual last.pt path:

```python
# ckpt is the Path to last.pt in your persisted run directory.
import subprocess, sys
subprocess.run([sys.executable, 'scripts/06_train.py', '--config', 'configs/default.yaml',
                '--backbone', 'efficientnet', '--frozen', '--resume', str(ckpt)], check=True)
```

Do not change the epoch schedule when resuming. A completed run is not silently
overwritten; use a fresh output root for a new experiment. Training stops after
an epoch once the configured time budget is reached; allow margin for long epochs
and calibration. A process killed mid-epoch resumes from the last completed one.

## 7. Evaluate each checkpoint without fine-tuning

The default checkpoint is the tiny calibrated run. For real backbones, pass the
actual calibrated checkpoint Path. All artifacts are under output_root/results.

```python
!python scripts/07_evaluate.py --config configs/default.yaml
# ckpt is the Path to best_calibrated.pt for the backbone/mode being evaluated.
subprocess.run([sys.executable, 'scripts/07_evaluate.py', '--config', 'configs/default.yaml',
                '--checkpoint', str(ckpt)], check=True)
```

Evaluation saves predictions, per-source metrics, per-class F1, confusion matrices,
ECE/AUC, all nine missingness modes, robustness plots, group-bootstrap intervals,
and Grad-CAM/SHAP for up to 10 correct and 10 wrong in-domain test examples.
Unavailable outcomes are reported, not manufactured. Backbone comparison CSV
accumulates completed evaluations; it does not fabricate unrun backbone results.
`--no-explainability` is available for diagnosis only; full paper evaluation should
include explanations. No external data enters training or calibration.
