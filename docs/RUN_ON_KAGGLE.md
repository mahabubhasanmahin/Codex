# Run on Kaggle

Enable a single GPU accelerator for training. Internet is needed only to install
packages, clone the repository and fetch explicitly requested pretrained weights.
Attach the five datasets at the paths in `configs/default.yaml`.

```python
!git clone https://github.com/mahabubhasanmahin/Codex.git
%cd Codex
!python -m pip install -r requirements.txt
```

```python
!python scripts/01_merge.py --config configs/default.yaml --schemas configs/dataset_schemas.yaml
!python scripts/02_labels.py inventory --config configs/default.yaml
```

Review `configs/label_mapping.csv`: supply a unified label and coarse group for
included rows, or mark them False with a reason. No diagnosis mappings are inferred.

```python
!python scripts/02_labels.py apply --config configs/default.yaml
!python scripts/03_metadata.py inventory --config configs/default.yaml
```

Review the generated site mapping and fill every observed raw site value.

```python
!python scripts/03_metadata.py apply --config configs/default.yaml
!python scripts/04_dedup.py --config configs/default.yaml
!python scripts/05_build_unified.py --config configs/default.yaml
!python scripts/06_train.py --config configs/default.yaml --backbone tiny --epochs 1
!python scripts/07_evaluate.py --config configs/default.yaml
```

Use a configured pretrained backbone instead of `tiny` for thesis experiments.
Keep zero-shot data separate throughout. All generated data/results live below
`paths.output_root`; editable mapping configuration is the explicit exception.
