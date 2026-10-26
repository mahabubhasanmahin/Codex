# CrossSkinNet project rules

- Write COMPLETE files. No placeholders, no "...", no pseudo-code, no "TODO: implement". The only allowed TODOs are data-specific mappings that I must fill from my real data, and only inside YAML/CSV config files, each with a comment saying exactly what to fill.
- Python 3.10+, type hints, docstrings, small functions, seed=42 read from config.
- The code runs on Kaggle notebooks (PyTorch, single GPU, 12-hour session limit). All outputs go to paths.output_root in configs/default.yaml (default /kaggle/working). No hardcoded paths inside code.
- Never guess folder structure or column names. They come from configs/dataset_schemas.yaml. If something is unknown, fail with a clear error message that names what is missing.
- Tests: no internet and no pretrained-weight downloads. Use tiny synthetic data and a tiny random backbone. Run `pytest -q` after every step and commit after every step with message "step N: <name>".
- Every script in scripts/ is a thin CLI over src/crossskinnet/. Print shapes, head and counts, and save intermediate results as CSV.
- Missing metadata is never imputed with mean or mode. Keep NaN and add a mask column per field.
- DATA RULES (never break them): user-approved revised training sources = PAD-UFES-20 + disease-label-reviewed Dermnet. Current overlapping ISIC2018 is disabled in default config; do not re-enable it for the revised study. Synthetic regression fixtures may explicitly enable the legacy source to test overlap exclusion. Zero-shot test only = HAM10000 + PH2; they are never used for training or validation. Any training image that is duplicated in HAM10000 or PH2 is removed from the TRAINING pool. The split is patient/lesion-level, no group appears in two splits.
- Kaggle input paths: PAD-UFES-20=/kaggle/input/datasets/maxjen/pad-ufes-20, ISIC2018=/kaggle/input/datasets/mahabubhasanmahin/isic2018-final, Dermnet=/kaggle/input/datasets/shubhamgoel27/dermnet, HAM10000=/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000, PH2=/kaggle/input/datasets/spacesurfer/ph2-dataset

