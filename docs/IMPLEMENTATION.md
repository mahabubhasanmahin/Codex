# Paper-plan implementation and limits

The supplied PDF is a research plan, not a report of measured results. Slides
18–24 specify dual encoding, gated cross-attention, all-missing image fallback,
Grad-CAM/SHAP, held-out testing, zero-shot HAM/PH2 testing and backbone comparisons.
These are implemented; real-data training and conclusions remain outstanding.

## Scientific choices made explicit

- Development data: PAD-UFES-20, ISIC2018 and both original Dermnet splits.
  External data: HAM10000 and PH2 only; no training, calibration or tuning on them.
- The inspection inventory includes matching ISIC/HAM image IDs. Matching names
  alone are not proof of identical pixels. Runtime SHA256 and pHash determine
  overlap. If all ISIC images match external images, all those training rows are
  removed. Never reintroduce them to improve dataset size or metrics.
- Duplicate components are conservative: a transitive near-duplicate chain
  touching external data removes every training member. Equal pHashes are bucketed;
  the report stores representative connectivity edges, not every quadratic pair.
  SHA256 equality is reported separately from perceptual similarity. Review the grid.
- Training-only duplicate components keep the row with most observed model metadata.
  External duplicates are retained. HAM lesion groupings are exported for auditing.
- Split ratios approximate 70/15/15 because patient/lesion groups are indivisible.
  Stratified group folds adjust resolution to the number of groups per class.
  Classes with fewer than three independent groups fail rather than leak images.
  For global lesion identifiers, missing patient IDs are linked through a unique known patient where possible. Patient-scoped lesion numbers require a patient ID.
- Raw patient/lesion identifiers are retained; collisions across sources are
  conservatively grouped. Dermnet has no inspected patient/lesion metadata; its
  image-ID fallback cannot establish biological patient independence. This limits
  claims of patient-independent evaluation for that source.
- Missing values stay NaN/unknown in tables, with observed masks, missing flags and
  structural flags. Numeric zero fillers exist only in model inputs; masks and
  learned per-field tokens distinguish missingness. Age is divided by configured
  age_scale, not normalized with population means.
- Full-metadata baseline means all *observed available* fields, not invented
  metadata for Dermnet or ISIC. Ablations never fill a structurally missing field.
- Image-only logits get auxiliary supervision and validation-only temperature
  scaling. The fusion head is not temperature-scaled by that scalar. Checkpoints
  store seeds/RNG states, optimizer, scaler and scheduler; resume requires the
  original dataset, run directory and hyperparameter schedule.
- The tiny CNN is exclusively for software validation. Real comparisons use the
  configured four backbones with identical split/seed/preprocessing/hyperparameters,
  with a lower backbone learning rate. Frozen backbone statistics remain fixed.
- SHAP explains semantic-field presence conditioned on a fixed image. Background
  is that same image with every field absent; the gate transition is included.
  Plots summarize selected correct/wrong examples, not a population causal effect.
- Bootstrap intervals resample patient/lesion groups, retaining their images.
  Macro-F1 uses a fixed training-class vocabulary; ROC AUC averages only defined
  one-vs-rest classes and reports how many were defined. Missing metrics are null.
  A single seed's bootstrap interval does not measure training-seed variability.

## What has and has not been validated

The automated suite uses tiny generated data with no network requests or weight
retrieval. It covers both complete CLI flow and local tiny random DINO adapters.
Random EfficientNet/Swin forward and Grad-CAM paths were also checked locally.
No pretrained real-data experiment, CUDA training, gated Hugging Face access or
full 50k-image timing benchmark was run in this workspace. No thesis accuracy is
claimed. Checkpoint files must be trusted local artifacts (PyTorch pickle loading).

## Data-specific review still needed

1. `configs/label_mapping.csv`: 46 observed source labels have intentionally empty
   unified labels/coarse groups. Define the thesis taxonomy with domain review;
   broad Dermnet folder names must not be silently treated as a single diagnosis.
2. Run site inventory on the real merged data; populate null entries in
   `configs/site_mapping.yaml` with reviewed anatomical groups.
3. Run ingestion on the actual Kaggle mounts. The archive supplies inventories
   and samples, not complete data. PH2 validates headers and clinical markers at
   runtime; it does not guess rows or read lesion/ROI masks as skin images.
4. Confirm licenses before uploading derived images, and inspect duplicate pairs
   and post-dedup class/source counts before launching real experiments.

See AUDIT_FINDINGS.md for measured ISIC overlap evidence and PAD identifier scope from the uploaded reports.
