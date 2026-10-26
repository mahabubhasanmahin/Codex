# Recommended complete workflow

Use notebooks/03_complete_paper.ipynb and docs/PAPER_WORKFLOW.md for the single
entry-point workflow. It uses an explicit group-quarantine cohort policy, not
manual pHash guesses. The instructions below preserve the individual-stage
human-review workflow for reproducibility; they are not required to click through
when using configs/paper.yaml.

# Current Kaggle workflow: PAD + HAM development, PH2 external

Use PAD-UFES-20 and HAM10000 for development. PH2 is the external-only test and
overlap reference. Dermnet is optional/disabled pending disease-label review;
ISIC2018 is disabled and not required. Collection membership does not attach data:
use **Add Input** in the notebook to attach the three active datasets.

This changes the original PDF's data roles. See REVISED_STUDY.md. Historical
HAM-external audit counts/checkpoints are not current-protocol results.

## 0. Code, dependencies and optional synthetic smoke

```python
!git clone https://github.com/mahabubhasanmahin/Codex.git
%cd Codex
!python -m pip install -r requirements.txt
!python -m pytest -q
```

If the checkout already exists, enter its actual directory and use
`git pull --ff-only origin main` instead of cloning again. Preserve any local
config/mapping edits if Git reports a conflict; do not reset them blindly.
Internet is needed for installation, cloning and uncached pretrained weights.
Tests and the tiny demo do not request pretrained weights.

```python
!python scripts/00_smoke.py --config configs/default.yaml
```

The demo writes a fresh SYNTHETIC_DEMO folder under configured output_root,
including all stages, tiny training, metrics, plots, Grad-CAM and SHAP. It uses
PAD patient proxies, HAM lesion proxies and PH2 all-missing metadata. A synthetic
HAM/PH2 duplicate checks external priority. Its metrics are not thesis results.

## 1. Inspect mounts and merge real data

```python
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
for name, settings in cfg['datasets'].items():
    if settings.get('enabled', True):
        print(name, settings['pool'], Path(settings['path']).exists(), settings['path'])
```

If a path is absent, inspect the actual Kaggle mount and edit its config value;
do not guess a folder layout or change disease column mappings to bypass an error.

```python
!python scripts/01_merge.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

Expected raw_train_pool.csv sources: PAD-UFES-20 and HAM10000.
Expected raw_test_pool.csv source: PH2 only. Check source_selection.csv/json.
For a pre-taxonomy feasibility audit instead, the following refreshes merge too:

```python
!python scripts/00_audit.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

Share output_root/audit/audit_reports.zip to review aggregate counts and metadata.
Audit pools/counts are provisional when relevant perceptual decisions are pending;
they do not approve a disease taxonomy or substitute for stages 02–05.

## 2. Review clinical labels

The committed mapping already selects the three-class core. Inventory preserves
it. Blank fields on include=False rows do not need invented labels.

```python
!python scripts/02_labels.py inventory --config configs/default.yaml
```

Fill unified_label/coarse_group for included active source labels in
configs/label_mapping.csv, or set include=False with a reason. Inventory preserves
edits. Inactive ISIC/Dermnet mapping rows do not need invented values.
SCC is not AK/AKIEC; PAD SEK is not automatically the entire HAM BKL category.
PH2 common/atypical-nevus equivalence must be explicitly reviewed.

```python
!python scripts/02_labels.py apply --config configs/default.yaml
```

External classes without development counterparts are flagged. All raw PH2 images
still remain duplicate references, including excluded external disease labels.

## 3. Normalize metadata after site review

```python
!python scripts/03_metadata.py inventory --config configs/default.yaml
```

Map every null raw anatomical site in configs/site_mapping.yaml to one allowed
group: head_neck, trunk, upper_limb, lower_limb, hands_feet or other.

```python
!python scripts/03_metadata.py apply --config configs/default.yaml
```

Age/sex/site are not statistically imputed. Symptoms absent in HAM and all model
metadata absent in inspected PH2 remain structurally missing with explicit masks.

## 4. Verify duplicate candidates

```python
!python scripts/04_dedup.py --config configs/default.yaml
```

SHA256-equal images are grouped automatically. pHash radius (default 5) generates
candidates only, even at distance zero. Review each relevant candidate in
output_root/near_duplicate_reviews.csv: set is_duplicate to True/False with notes;
leave uncertain pairs blank. Do not mark all candidates True/False to bypass review.
Decisions are bound to content hashes and are reused by audits and stage 04.

PH2-overlap removes the DEVELOPMENT copy, never PH2. Verified PAD/HAM duplicates
are internal pool duplicates and keep the best-metadata copy. Unresolved relevant
pairs or conflicting decisions intentionally block readiness. Inspect the review
grid and full candidate/report CSV; a grid samples at most 20 pairs.

```python
!python scripts/00_review_overlap.py --config configs/default.yaml
```

That helper reads existing audit reports and creates focused review outputs; it
is optional and does not approve any candidate. After decisions, rerun stage 04.
Its dedup_status.json must report ready=true before stage 05. Old pool files cannot
bypass a new pending status. A changed review file requires stage 04 again.

## 5. Export images and group splits

```python
!python scripts/05_build_unified.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

PAD requires patient_id; HAM requires lesion_id. Missing required identifiers fail
rather than silently splitting by image. Approximate train/val/test = 70/15/15;
unequal groups limit exact ratios. HAM is lesion-disjoint, not necessarily patient-
disjoint across different lesions when patient IDs are unavailable.

Inspect CrossSkinNet_Unified_Dataset/metadata.csv, zero_shot.csv, grouping_schema.yaml,
dataset_stats.json and distribution plots. Both PAD/HAM must occur in the actual
training split. PH2 goes only to zero_shot.csv. The dataset integrity manifest binds
CSV contents; do not manually alter split rows after building.
See upload_to_kaggle.md for private reuse via paths.unified_root.

## 6. Train on a single GPU

First verify execution, not scientific performance:

```python
!python scripts/06_train.py --config configs/default.yaml --backbone tiny --epochs 1
```

Real example (configured 30-epoch schedule, time-budget stop and early stopping):

```python
!python scripts/06_train.py --config configs/default.yaml --backbone efficientnet --frozen
!python scripts/06_train.py --config configs/default.yaml --backbone efficientnet --no-frozen
```

Repeat for swin, dinov2 and dinov3 with the identical prepared split. Model IDs are
in YAML. Accept gated DINOv3 license terms and provide HF_TOKEN securely through
Kaggle secrets/environment; never commit it. Keep torch/torchvision compatible.
Pretrained/CUDA performance is not validated by the offline tiny tests.

Use one backbone/mode per session if needed. Keep its whole run directory;
checkpoints save every completed epoch. Resume with the same config, backbone,
frozen flag and epoch schedule plus `--resume` pointing to the actual last.pt.
The configured time limit is checked after an epoch, so allow a margin for epoch
length and calibration. Completed runs are not silently overwritten.

For multiple training seeds, prepare the split once. Reuse the same paths.unified_root,
vary seed in separate configs, and use distinct output_root values. Do not regenerate
a different dataset split per backbone. This is not an automatic multi-seed runner.

## 7. Evaluate without fine-tuning

```python
!python scripts/07_evaluate.py --config configs/default.yaml
```

Default checkpoint: the tiny calibrated run. For real models pass the actual
best_calibrated.pt using `--checkpoint`. Results go below output_root/results.
PAD/HAM held-out test results are IN-DOMAIN. PH2 alone is external and evaluated
only for labels present in the trained vocabulary. Its inspected fields are all
missing, so it uses the validation-calibrated image-only fallback.

Artifacts include per-source metrics/predictions, explicit-scope macro-F1 (plus a full-vocabulary secondary score),
per-class F1, confusion matrices, defined-class AUC, ECE, all nine ablation modes,
group-bootstrap intervals and up to 10 correct/10 wrong Grad-CAM+SHAP examples.
Unavailable outcomes are reported. SHAP uses field-presence coalitions conditioned
on each image; it is not proof of causality. Bootstrap intervals do not capture
training-seed uncertainty. Backbone comparison accumulates only completed runs.

Do not call pooled HAM results zero-shot. PAD-only→HAM or HAM-only→PAD require
separate study configurations, overlap checks and training runs. They are not
performed by the default pooled pipeline.

## Current three-class mapping and one-pair-at-a-time review

The current mapping selects melanoma, basal_cell_carcinoma and melanocytic_nevus.
PH2 Common Nevus/Melanoma are included; Atypical Nevus is conservatively excluded
from evaluation but retained in ALL raw PH2 overlap references. See
THREE_CLASS_PROTOCOL.md for the complete mapping, exclusions and provenance limits.

Use notebooks/02_three_class_review.ipynb to bootstrap, merge, prepare selected-label
candidates and review them with explicit True/False/Unsure buttons. ipywidgets is
needed only for the interactive notebook UI and is included in requirements.txt.
This is not a new heavy model dependency. The preparation CLI does not need site
mappings or pretrained weights and must not be used as a normalized training pool.
