# CrossSkinNet

Explainable Multimodal Skin Disease Classification Under Missing Metadata.
Python 3.10+, targeting Kaggle notebooks with one GPU and a 12-hour session limit.

Steps 0–2 are implemented. Further implementation is paused for reviewed medical
label mappings, following the requested stop-on-missing-information rule.
See [the Kaggle notebook cells](docs/RUN_ON_KAGGLE.md) and [project rules](AGENTS.md).

```sh
python -m pip install -r requirements.txt
pytest -q
python scripts/01_merge.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
python scripts/02_labels.py inventory --config configs/default.yaml
# Review configs/label_mapping.csv before applying.
python scripts/02_labels.py apply --config configs/default.yaml
```

## Configuration and data safety

`configs/default.yaml` contains the seed (42), all dataset roots, pool membership
and `paths.output_root` (default `/kaggle/working`). All runtime outputs use this
root. The editable label mapping in `configs/` is the requested exception.

Training sources are PAD-UFES-20, ISIC2018 and Dermnet. HAM10000 and PH2 are
external test only. Step 1 does not split or deduplicate: raw pools must not be
used for training until external-overlap removal and group splitting are built.
Missing metadata remains NaN plus a per-field observed mask (1 observed, 0 missing).
No statistical imputation or automatic diagnosis equivalences are used.

## Implemented stages

Step 1 writes separate `raw_train_pool.csv` and `raw_test_pool.csv`, printing shapes,
heads and source/label counts. The dataset schemas come from `dataset_inspection.zip`:

- PAD: metadata diagnoses, patient/lesion identifiers and six symptoms.
- ISIC: one-hot labels joined with image-to-lesion groupings.
- Dermnet: class folders from both original train and test partitions.
- HAM: metadata diagnoses and uppercase image directories, selecting one copy
  of the duplicate upper/lowercase collections in the inventory.
- PH2: clinical X markers from `Folha1`, located by inspected headers; only
  dermoscopic BMPs are ingested, excluding lesion/ROI masks.

Schema mismatches, missing files, ambiguous IDs and invalid one-hot labels fail
explicitly. PAD `UNK` and HAM `unknown` metadata values become NaN. Relative data
and output paths resolve against the configuration file directory. The real Kaggle
files are not available here, so real ingestion remains to be verified on Kaggle.
The inspection reports PH2 column samples rather than complete worksheet rows;
its reader verifies headers and one clinical label per image at runtime.
`openpyxl` is needed for PH2 spreadsheets; no model dependencies are added yet.

Step 2 inventory prints exact observed label counts and writes
`original_label_counts.csv`, adding missing entries to the editable mapping without
overwriting existing decisions. The committed mapping contains 46 observed labels
with unified labels and coarse groups deliberately blank. Supply reviewed values
or set `include=False` with a reason. Applying an unfinished mapping fails.

Apply writes `labeled_train_pool.csv`, `labeled_test_pool.csv`, `label_counts.csv`,
`label_exclusions.csv`, and `label_warnings.csv`. Explicit exclusions are counted
and logged. External classes without a training counterpart remain in the external
table with `zero_shot_eligible=False` and appear in the warnings report. No rows
are silently dropped. Warnings also list training classes with fewer than 50 rows.

## Offline fixtures and verification

`tests/fixtures/make_synthetic_data.py` creates 30 tiny random RGB images per source,
metadata CSVs, and synthetic config/schema files in a temporary directory. Each
source has an internal duplicate, and ISIC/HAM share an image. These layouts make
no claims about the real datasets. Tests own cleanup; direct callers must remove
the returned temporary root themselves. Additional miniature layouts test the
production schemas, including a PH2 spreadsheet and mask exclusion.
Tests require no network or pretrained weights. Future model tests must use a
small random backbone. `AGENTS.md` records the masking,
data isolation and per-step test/commit requirements.
