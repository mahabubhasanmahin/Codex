# CrossSkinNet

Explainable Multimodal Skin Disease Classification Under Missing Metadata.
Python 3.10+; intended runtime: Kaggle notebooks, PyTorch, one GPU, 12-hour sessions.
This repository contains schema-driven raw dataset ingestion, later-stage placeholders, and offline test fixtures.
PyTorch is not required by the skeleton; no weights or real datasets are downloaded.

```sh
python -m pip install -r requirements.txt
pytest -q
python scripts/01_merge.py --help
```

`configs/default.yaml` defines seed 42, input paths and the output root
(default `/kaggle/working`). All pipeline outputs must use `paths.output_root`.
Actual dataset layouts and column mappings must be inspected and configured before
implementing ingestion; these are deliberately unspecified here.

Pipeline stages: `01_merge` (implemented), `02_deduplicate`, `03_split`, `04_preprocess`,
`05_train`, `06_explain`, `07_evaluate`. Scripts are thin CLIs over
`src/crossskinnet/`; each accepts `--config`. Stages 02–07 remain stubs.

Step 1 writes `raw_train_pool.csv` and `raw_test_pool.csv` under the configured
output root and prints shapes, sample rows and source/label counts. It preserves
raw duplicates for the later deduplication stage; these tables are not ready
for training until external-overlap removal and group splitting are complete.

```sh
python scripts/01_merge.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
```

`configs/dataset_schemas.yaml` is derived from the uploaded
`dataset_inspection.zip` inventories and column samples:

- PAD: metadata CSV, `diagnostic` labels, patient/lesion IDs and six symptoms.
- ISIC: one-hot ground truth plus the image-to-lesion grouping CSV.
- Dermnet: class folder labels from both original train and test directories.
- HAM: metadata diagnoses and uppercase image directories, selecting one copy
  of the duplicated upper/lowercase collections shown in the inventory.
- PH2: `Folha1` spreadsheet clinical diagnosis X markers, located via inspected
  header text; only dermoscopic BMPs are selected, excluding lesion/ROI masks.

The PH2 archive reports column samples rather than complete worksheet rows.
The reader discovers header columns at runtime and requires exactly one clinical
label per image; inconsistent layouts fail rather than assigning guessed labels.
The real Kaggle files are unavailable in this workspace, so full ingestion must
still be verified there. Tests exercise these schemas on miniature local layouts.
`openpyxl` reads PH2 XLSX files without introducing pandas or model dependencies.

The CSV loader supports explicitly listed metadata partitions, column/symptom
mappings, filename templates, indexed image globs, one-hot labels and a validated
lesion join. Missing columns/images, duplicate image IDs in configured directories,
and ambiguous labels produce errors. Image paths resolve against each configured
dataset root. Relative dataset/output roots resolve against the config directory.
PAD `UNK` and HAM `unknown` metadata values become NaN with a missing mask.

Synthetic fixtures generate their own `dataset_schemas.yaml`. Use that file and
the generated `config.yaml` with the same CLI to exercise ingestion offline.
Missing fields are serialized as `NaN`; metadata masks use 1 for observed and 0
for missing. No metadata is imputed.

PAD-UFES-20, ISIC2018 and Dermnet form the training pool. HAM10000 and PH2
are zero-shot external tests only: never train or validate on them. Remove
training images duplicated in either external dataset before splitting.
Split by patient/lesion `group_id`, with no group shared between splits.
Preserve missing metadata as NaN plus a per-field mask; never mean/mode impute.
Future pipeline stages must print shapes, heads and counts and save intermediate CSVs.

`tests/fixtures/make_synthetic_data.py` generates a temporary synthetic root with
30 random 16×16 RGB PNGs per dataset, CSV metadata and a generated config.
Its synthetic layout is not a claim about the real Kaggle data. Each dataset
contains an internal duplicate; ISIC2018 and HAM10000 also share an image.
`age_mask=1` means observed and `0` means missing. Tests own cleanup; callers
using the generator directly must remove the returned temporary directory.
Tests stay offline and use synthetic data; future model tests must use a tiny
random backbone. No pretrained-weight downloads are permitted in tests.
