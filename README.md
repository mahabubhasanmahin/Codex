# CrossSkinNet

Explainable Multimodal Skin Disease Classification Under Missing Metadata.
Python 3.10+, targeting Kaggle notebooks, PyTorch, a single GPU and 12-hour sessions.
The code implements stages 0–7. It has been exercised on synthetic data, not trained
on the real medical datasets. No experimental accuracy is claimed.

**Start here:** [Kaggle notebook](notebooks/CrossSkinNet_Kaggle.ipynb) and
[exact commands](docs/RUN_ON_KAGGLE.md).

```sh
python -m pip install -r requirements.txt
pytest -q
python scripts/00_smoke.py --config configs/default.yaml
```

The demo uses tiny random images/backbones, test-only label mappings, and no weight
downloads. It writes a new SYNTHETIC_DEMO directory under configured output_root,
with all stage outputs, a checkpoint, metrics, plots, Grad-CAM and SHAP examples.
These demo metrics are software checks and must never appear as thesis results.

The real-data workflow is merge → explicit label harmonization → missing metadata
normalization → cached SHA/pHash duplicate review → group split/image export → model
training/calibration → evaluation. Every stage has a thin numbered CLI in scripts/.

`configs/default.yaml` holds seed 42, dataset paths, output root and hyperparameters.
`configs/dataset_schemas.yaml` derives from the uploaded inspection archive.
`configs/label_mapping.csv` records the selected three-class operational codebook;
`configs/site_mapping.yaml` still needs anatomical mappings. Source annotations
are not re-diagnosed, and unknown clinical equivalences are not invented. Their inventory commands preserve edits and print outstanding entries.

Current development sources: PAD-UFES-20 and HAM10000.
PH2 is external-only. Dermnet is optional and disabled pending disease-label review;
current ISIC2018 is disabled. Only PAD, HAM and PH2 must be attached.
Verified duplicates touching external data are removed from training, never from
external pools. pHash candidates require explicit review; hash similarity alone
does not remove images. Pending relevant reviews block dataset build/training. Matching ISIC/HAM IDs in the inventory signal potentially severe overlap;
only runtime image hashes establish exact/perceptual duplicates. Groups never
cross development splits. Unknown metadata stays missing; model zero fillers are
masked and replaced by learned missing-field tokens.

Models support a tiny offline CNN, timm EfficientNet/Swin and Hugging Face DINOv2/
DINOv3. Available metadata uses gated cross-attention; all-missing records use a
validation-calibrated image head. Training includes metadata dropout, focal loss,
class weights, lower backbone LR, AdamW/cosine scheduling, AMP, early stopping and
resumable epoch checkpoints. Evaluation covers held-out/per-source and zero-shot
metrics, ablations, explanations and group-bootstrap backbone-comparison intervals.
Real pretrained/CUDA experiments and Kaggle data ingestion still need validation.

Read [implementation choices and scientific limits](docs/IMPLEMENTATION.md),
[private dataset upload instructions](docs/upload_to_kaggle.md),
[file inventory](docs/FILES.md) and [agent rules](AGENTS.md).

Before choosing real disease classes, run the mapping-free feasibility audit:

```sh
python scripts/00_audit.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

It refreshes merge and writes label-review, metadata availability and post-dedup
counts under output_root/audit, retaining all external images as overlap references.
No medical equivalences are inferred and no training is started. See the Kaggle
guide for displaying/sharing the reports; these raw audit pools are not training-ready.


The user-approved PAD/HAM + PH2 revision is documented in docs/REVISED_STUDY.md.
`enabled: false` skips a development dataset before accessing its filesystem.
PH2 must remain enabled/external; current required development sources are PAD/HAM. Merge saves
source_selection.csv/json, downstream stages reject stale disabled-source rows,
and the unified dataset README/statistics name the sources actually retained.
Enabling a source is not clinical approval of its labels: unknown mappings and
pending duplicate decisions still block training readiness.

Current shared-class task: melanoma, basal_cell_carcinoma, melanocytic_nevus.
See docs/THREE_CLASS_PROTOCOL.md for exact source mappings and deliberate exclusions.
For selected-data duplicate review before site normalization, open
notebooks/02_three_class_review.ipynb. Its explicit per-pair decisions save to CSV;
no candidate is automatically approved or rejected.
