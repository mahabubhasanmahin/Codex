# CrossSkinNet: complete current workflow

Use `notebooks/03_complete_paper.ipynb`. The current study is PAD+HAM development,
PH2 external, optional Dermnet disabled. The title does not imply all skin diseases:
the core task is melanoma / basal cell carcinoma / melanocytic nevus. Source
annotations and the selected clinical codebook are recorded, not newly diagnosed.

## What is implemented

- Inspected dataset ingestion, explicit three-class mapping/exclusion counts and
  anatomy mapping from observed PAD/HAM strings; unseen anatomy fails explicitly.
- NaN/missing/structural masks, learned field tokens, random metadata dropout and
  image-only fallback. No mean/mode imputation.
- SHA-exact and accepted duplicate grouping, indexed pHash candidate generation,
  and an opt-in conservative **group-quarantine** cohort policy.
- Fixed PAD-patient / HAM-lesion group splits and portable image/metadata export.
- CrossSkinNet gated cross-attention plus controlled image-only and concatenation
  baselines. Concat shares the metadata encoder/missing tokens/fallback so it is
  a controlled fusion-mechanism comparison, not a reproduction of MetaNet/MetaBlock.
- EfficientNet, Swin, DINOv2, DINOv3 adapters, frozen/fine-tuned modes, class-weighted
  focal loss, AdamW/cosine LR, CUDA AMP, early stopping and resumable epoch snapshots.
- Validation-only fallback temperature calibration; in-domain/per-source, PH2
  external, nine field/missingness ablations, ROC AUC/ECE/confusion matrices,
  group-bootstrap intervals and Grad-CAM/field-presence SHAP outputs.
- A 30-experiment plan: 4 CrossSkinNet backbones x 2 modes x 3 seeds, plus 2
  EfficientNet-based baselines x 3 seeds. It is a plan, not a claim of execution.
  Experiments reuse the SAME prepared split, varying training seed only.
- Actual-result collection and seed mean/std tables. Tiny runs are software checks
  and are separated from paper_results.csv. Missing/unrun results are not invented.

## Why no 429-pair guessing task is required

`configs/paper.yaml` explicitly opts into quarantine; `configs/default.yaml` keeps
strict human-review mode for reproducibility. Under quarantine:

1. pHash never declares a pair duplicate/different. Existing explicit decisions
   can still be supplied, but no old review CSV is silently imported by this config.
2. Every unresolved candidate's recorded DEVELOPMENT groups are excluded before
   splitting. This closes over patient/lesion membership and verified duplicate
   links, so another image of the same quarantined group cannot re-enter the cohort.
3. Groups linked to verified external copies are excluded conservatively too.
   PH2 images are never deleted/quarantined; all 200 remain overlap references,
   including the 80 atypical-nevus images excluded from the evaluation label task.
4. `quarantined_training.csv` lists exclusions separately from
   `removed_training_duplicates.csv`. These are **not confirmed duplicate counts**.
   Source images are untouched. `cohort_flow.csv` accounts for every selected image.
5. The run fails if this leaves insufficient classes/groups or removes a required
   training source. It never loosens the threshold to recover performance.

This is a transparent cohort-selection policy, not an automatic clinical/image
identity oracle. It may exclude genuine distinct images and bias the retained
cohort. Disclose the threshold, policy and class/source/group exclusion counts in
the manuscript. Quarantine does not prove every possible cross-dataset linkage
was detected. HAM lesion disjointness is not patient independence across lesions.
Optional Dermnet image-ID grouping cannot establish patient independence either.

## Commands in repository root

```sh
python scripts/08_paper.py prepare --config configs/paper.yaml
python scripts/08_paper.py plan --config configs/paper.yaml
python scripts/08_paper.py train --config configs/paper.yaml --backbone efficientnet
python scripts/08_paper.py evaluate --config configs/paper.yaml --backbone efficientnet
```

Or one bounded experiment from preparation through explanations:

```sh
python scripts/08_paper.py run --config configs/paper.yaml --backbone efficientnet
```

Run one backbone/mode/seed per Kaggle session as needed. Use `--frozen` for frozen
runs, `--architecture image_only` or `--architecture concat` for baselines, and
`--seed 42`, `43` or `44`. The default is fine-tuned CrossSkinNet with seed 42.
Do not pass `--epochs 1` to a real experiment intended to use the configured schedule.

```sh
python scripts/08_paper.py train --config configs/paper.yaml --architecture image_only --backbone efficientnet --seed 42
python scripts/08_paper.py train --config configs/paper.yaml --architecture concat --backbone efficientnet --seed 42
python scripts/08_paper.py train --config configs/paper.yaml --backbone swin --frozen --seed 43
```

Evaluate with the SAME architecture/backbone/seed/frozen flags. The runner resumes
an incomplete existing run automatically from last.pt, and reuses a completed one
only when config/dataset match and calibration completed. A time-budget stop is
not a completed experiment: retain the whole output directory and resume next
session; final evaluation refuses incomplete runs. Changing schedule/config on
an existing experiment requires a fresh output root. Preparing a changed dataset
also refuses reuse of old experiment checkpoints.

A synthetic/tiny diagnostic run uses `--backbone tiny --epochs 1`. Its results are
kept in software_smoke_results.csv, not the paper table. All offline tests use
synthetic data and random weights; no pretrained download is performed by tests.

## Outputs under configured paper output_root

- `COHORT_REPORT.md`, `cohort_summary.json`, `cohort_flow.csv`: real preparation
  counts, including quarantine as a separate exclusion, not a duplicate verdict.
- `configuration_snapshot/`: exact config, inspected schemas, clinical/site maps.
- `CrossSkinNet_Unified_Dataset/`: fixed CSVs, 256x256 JPEGs, group schema, integrity
  manifest, stats and distribution plots. Unchanged image export can be reused.
- `experiment_plan.csv`: all requested experiments; this does not launch them.
- `experiments/<architecture>_<backbone>_<mode>_seed<N>/`: run config, epoch
  checkpoints, optimizer/scaler/RNG state, calibration, metrics and evaluation.
- `results/paper_results.csv`: only actual completed, evaluated real-backbone runs.
- `results/software_smoke_results.csv`: tiny random-backbone checks, never thesis evidence.
- `results/seed_summary.csv`: completed seed count, mean/std; one-seed std is null.
- Per-experiment results: per-source/class metrics, full/supported class scopes,
  confusion matrices, group bootstrap, robustness table/plot, Grad-CAM/SHAP.

Do not compare models on different prepared split fingerprints. Full-vocabulary
in-domain macro-F1 and PH2-supported-label external macro-F1 are explicitly scoped;
full-vocabulary external score is also retained. Predicting BCC on PH2 still counts
as an error. No external BCC validation is claimed because PH2 has no BCC samples.

## Real execution limits

This workspace has not run the Kaggle medical data or pretrained GPU experiments.
The software pipeline and baselines are tested with generated images and tiny
random models. DINOv3 can require model-license acceptance/HF_TOKEN; configure it
securely before that run. Internet is for installation/uncached weights, not tests.
The 12-hour budget is checked after epochs; allow a margin and persist the entire
run directory. Not every experiment is expected to finish in a single session.

The core implements the nine agreed field/missingness ablations. It does not claim
clinical MNAR modeling, MetaNet/MetaBlock reproduction, dermatologist review or
publication readiness. Such claims require a specified protocol/additional work,
real measurements and scientific review. Explanations are model attribution, not
proof of causality or lesion localization accuracy without lesion annotations.
