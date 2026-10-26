# Shared-three-class core protocol

The user selected a disease-level shared-class experiment after auditing PAD+HAM
training and PH2 external data. This file records the operational taxonomy, not
new image annotations or an independent clinician review of the source labels.
Unknown broader disease equivalences are not inferred.

|Source|Original label|Unified label|Coarse group|Included|
|---|---|---|---|---|
|PAD-UFES-20|MEL|melanoma|malignant|True|
|HAM10000|mel|melanoma|malignant|True|
|PH2|Melanoma|melanoma|malignant|True|
|PAD-UFES-20|BCC|basal_cell_carcinoma|malignant|True|
|HAM10000|bcc|basal_cell_carcinoma|malignant|True|
|PAD-UFES-20|NEV|melanocytic_nevus|benign|True|
|HAM10000|nv|melanocytic_nevus|benign|True|
|PH2|Common Nevus|melanocytic_nevus|benign|True|

PAD ACK/SCC/SEK and HAM akiec/bkl/df/vasc are excluded from this core task, not
mislabelled as one of the three targets. SCC is not AK/AKIEC. SEK is not the entire
BKL category. Dermnet/ISIC remain disabled. PH2 Atypical Nevus is conservatively
excluded from evaluation until an explicit extension of this taxonomy is reviewed.
All 200 raw PH2 images still protect against training overlap, including the
80 atypical-nevus images. Given the user's observed PH2 counts, the selected
external evaluation subset is 80 common nevi + 40 melanoma = 120 images.
PH2 provides no BCC samples; do not claim external BCC validation.

`study.target_labels` binds the task in config. Label apply, post-dedup build and
training fail if a target disappears or an unexpected class is introduced. Both
PAD and HAM must remain in the actual training split. Labels are not mean/mode
imputed or predicted from image appearance. All include=False removals are logged.
The mapping is editable, but changes invalidate a prepared review snapshot and
require preparation/normalization/dedup/build again.

## Source provenance and verification limits

- PAD descriptor/data page supplied in this session:
  https://data.mendeley.com/datasets/zr7vgbcyr2/1
- HAM10000 source descriptor:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/
- PH2 project page supplied in this session:
  https://www.fc.up.pt/addi/ph2%20database.html

Raw label spellings and counts were observed in the user's real inspection and
audit CSVs. Canonical MEL/BCC/NEV and mel/bcc/nv decoding is used for the explicitly
selected three-class protocol. Live access to all three source pages returned a
proxy 403 during this implementation, so these references are recorded as source
attributions, not falsely claimed as newly retrieved or independently verified.
Clinical equivalence and the deliberate exclusion of atypical nevi must be
reviewed/described in the manuscript; do not call this a dermatologist reannotation.

## Review only selected-label candidates

Use notebooks/02_three_class_review.ipynb or, from the repository root:

```python
!python scripts/04_prepare_review.py --config configs/default.yaml --mapping configs/label_mapping.csv
```

This applies the mapping, preserving all raw PH2 references, then previews dedup
in output_root/review_preparation. Existing validated hash caches are reused where
possible. It does NOT overwrite the main normalized metadata tables or train.
The result contains selection_summary.json, selected_class_counts.csv,
provisional_class_counts.csv, review_queue.csv and per-pair diagnostic reports.
No new real-data counts are claimed until this command runs on the Kaggle inputs.

```python
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd() / 'src'))
from IPython.display import display
from crossskinnet.data.review_ui import launch_review
panel = launch_review(Path('configs/default.yaml'))
display(panel)
```

Judge copies/transforms of the same image, not similar lesions or equal diagnoses.
Each decision button saves only that pair to the shared content-bound review CSV.
True joins duplicate components; False rejects that edge; Unsure leaves it pending.
Existing unrelated decisions are preserved. Previous/Next navigates without saving
notes. Pair decisions are explicit; no bulk auto-confirm/auto-reject function exists.
The panel displays a snapshot count, not a final readiness verdict. After reviewing,
rerun preparation to refresh components/relevance. Review conflicts must be resolved.
If widgets do not render, use the same review CSV manually or reopen the notebook;
do not bypass missing decisions by filling all rows automatically.

Site normalization (stage 03), actual stage 04, and stage 05 remain prerequisites
before real training. The preview config is explicitly blocked from building a
training dataset because its tables are still unnormalized. Review-ready does not
mean the complete scientific experiment is ready or that clinical labels were
independently validated.

## PH2 metric scope

PH2 has no BCC examples in this protocol. External primary macro-F1 averages the
supported evaluated labels (melanoma/nevus), and reports evaluated_classes explicitly.
The full three-class macro-F1 is retained as macro_f1_all_training_classes. Model
argmax is NOT restricted to the PH2 classes: predicting BCC on a PH2 example still
counts as wrong. Confusion matrices/per-class tables retain the full vocabulary;
an empty BCC truth row is not external BCC validation. Bootstrap intervals use the
same fixed external-supported label set as the point estimate, not a changing
class set in each resample. In-domain primary macro-F1 retains all training classes.
