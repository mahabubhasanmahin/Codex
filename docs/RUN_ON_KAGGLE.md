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
ISIC/HAM verified overlap may remove most/all ISIC training images; report that honestly.
If stage 04 reports pending pHash reviews, it is intentionally not ready. Review
near_duplicate_reviews.csv as described below, then rerun stage 04.
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

## Before choosing final disease classes: one-command feasibility audit

The audit requires no label or site mappings. It refreshes Step 1 from the actual
configured mounts (all five datasets are required), then hashes isolated raw
copies below output_root/audit. It does not train or overwrite normalized/labeled
pipeline tables. The first hashing pass can take time; subsequent audits reuse
fingerprint-validated hashes. This is raw-data feasibility, not clinical approval.

```python
!python scripts/00_audit.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

```python
from pathlib import Path
import pandas as pd
import yaml
cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
audit = Path(cfg['paths']['output_root']) / 'audit'
for name in ('source_summary', 'class_feasibility', 'metadata_availability', 'dermnet_review'):
    print('\n' + name)
    display(pd.read_csv(audit / f'{name}.csv'))
```

Share REPORT.md and these four CSVs to review the final disease taxonomy. Group
counts are recorded identifier counts/proxies, not proof of biological independence.
Counts are per original label, not per unified disease until equivalences are
reviewed. Dermnet folder wording alone cannot approve an image-level diagnosis.

The audit also creates `audit/audit_reports.zip` with the six aggregate report
files (no image data or row-level raw metadata). Download it from Kaggle's Output
panel and share it for review.

## Follow-up: inspect exact evidence and non-ISIC perceptual overlap candidates

After the audit, retain its files and attached images, then run:

```python
!python scripts/00_review_overlap.py --config configs/default.yaml
```

This does not hash all images again. Inspect audit/overlap_evidence.csv and
PAD-UFES-20_overlap_review.png / Dermnet_overlap_review.png. The matching *_pairs.csv
files contain all relevant representative edges; grids sample up to 20 each.
SHA-equality counts are proven lower bounds from reported edges. Nonzero-distance
pHash chains do not establish byte-identical images. See AUDIT_FINDINGS.md.

Step 5 can take `--schemas configs/dataset_schemas.yaml` explicitly. Otherwise it
loads dataset_schemas.yaml beside its config if available. Real PAD grouping now
validates the uploaded patient/lesion filename prefix and uses patient-scoped
lesion numbers; bare repeated lesion numbers do not merge unrelated patients.


## Corrected duplicate policy after the targeted image review

The original automatic `distance <= 5` removal was too aggressive. Default
near_duplicate_policy is now `review`; the Hamming threshold still generates
candidates, not decisions. Even distance-zero pHashes can collide.

- SHA256-equal files are automatically grouped, independently of pHash buckets.
- Other candidates are written to `<output_root>/near_duplicate_reviews.csv`.
  Review the images, set `is_duplicate` to True or False, and record a reason in
  notes. Leave uncertain pairs blank. Do not mark all candidates False or True
  just to bypass the check. Decisions bind to SHA content, not filenames.
- True joins the duplicate component. False rejects that candidate edge. A False
  inside a component already joined by accepted edges is a conflicting decision
  and must be resolved. Blank relevant pairs prevent readiness.
- Audit runs successfully with pending reviews but writes **provisional** counts.
  `<output_root>/audit/provisional_train_pool.csv` is not a training dataset.
- Stage 04 must finish without pending/conflicting relevant pairs before stage 05.
  Old deduped files are not trusted when the current status is pending. Stage 05
  also checks the review/CSV fingerprints; training/evaluation validate the built
  integrity manifest. Rebuild any previously prepared dataset with the new policy.

To regenerate the raw feasibility report using the corrected policy:

```python
!python scripts/00_audit.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

Share audit/audit_reports.zip for revised provisional counts. For review, inspect
near_duplicate_reviews.csv plus audit/near_duplicate_review.png and the pair IDs
in audit/duplicates_report.csv. Grids only sample up to 20; every remaining relevant
candidate needs its own decision. Existing hash caches are reused if fingerprints
match. Input images and raw source metadata are never deleted by deduplication.

After completing decisions, rerun the audit for raw feasibility and stage 04 for
the harmonized pipeline; reviewed metadata may alter which training copy is best.
The audit and stage 04 use the same default content-bound review file.

## Current approved setup: PAD + reviewed Dermnet

The current default excludes overlapping ISIC2018 (`enabled: false`). For the
revised study attach PAD, Dermnet, HAM10000 and PH2; the ISIC input is optional
and is not read. Earlier five-input/three-development-source descriptions above
refer to the original plan and its historical audit. See REVISED_STUDY.md.

After pulling, rerun merge or 00_audit. Inspect source_selection.csv to confirm
the four active inputs and ISIC exclusion. Do not restore its rows from an old
CSV. Label/site review and relevant pHash decisions remain prerequisites to real
training. The synthetic smoke command now mirrors the enabled-source selection;
legacy five-source fixtures remain available for regression checks.
