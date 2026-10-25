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
normalization → cached pHash duplicate removal → group split/image export → model
training/calibration → evaluation. Every stage has a thin numbered CLI in scripts/.

`configs/default.yaml` holds seed 42, dataset paths, output root and hyperparameters.
`configs/dataset_schemas.yaml` derives from the uploaded inspection archive.
`configs/label_mapping.csv` and `configs/site_mapping.yaml` require reviewed medical
and anatomical decisions before real-data training; no diagnosis equivalences are
invented. Their inventory commands preserve edits and print outstanding entries.

Training sources: PAD-UFES-20, ISIC2018, Dermnet. External only: HAM10000 and PH2.
Duplicates touching external data are removed from training, never from external
pools. Matching ISIC/HAM IDs in the inventory signal potentially severe overlap;
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
