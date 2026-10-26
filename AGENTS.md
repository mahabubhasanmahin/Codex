# CrossSkinNet project rules

- Write COMPLETE files. No placeholders, no "...", no pseudo-code, no "TODO: implement". The only allowed TODOs are data-specific mappings that I must fill from my real data, and only inside YAML/CSV config files, each with a comment saying exactly what to fill.
- Python 3.10+, type hints, docstrings, small functions, seed=42 read from config.
- The code runs on Kaggle notebooks (PyTorch, single GPU, 12-hour session limit). All outputs go to paths.output_root in configs/default.yaml (default /kaggle/working). No hardcoded paths inside code.
- Never guess folder structure or column names. They come from configs/dataset_schemas.yaml. If something is unknown, fail with a clear error message that names what is missing.
- Tests: no internet and no pretrained-weight downloads. Use tiny synthetic data and a tiny random backbone. Run `pytest -q` after every step and commit after every step with message "step N: <name>".
- Every script in scripts/ is a thin CLI over src/crossskinnet/. Print shapes, head and counts, and save intermediate results as CSV.
- Missing metadata is never imputed with mean or mode. Keep NaN and add a mask column per field.
- DATA RULES (current user-approved protocol): development = PAD-UFES-20 + HAM10000, with train/validation/in-domain-test group splits. PAD uses patient_id; HAM uses lesion_id (patient independence cannot be claimed when patient IDs are unavailable). PH2 is external-only, never train/validation/calibration. Any development image duplicated in PH2 is removed from development. Dermnet is optional and disabled until disease-label review; current ISIC2018 is disabled. Historical synthetic regression fixtures may explicitly configure older roles, but never mix their results with this protocol. Keep NaN/masks, review non-exact perceptual candidates and assert group disjointness.
- Kaggle input paths: PAD-UFES-20=/kaggle/input/datasets/maxjen/pad-ufes-20, ISIC2018=/kaggle/input/datasets/mahabubhasanmahin/isic2018-final, Dermnet=/kaggle/input/datasets/shubhamgoel27/dermnet, HAM10000=/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000, PH2=/kaggle/input/datasets/spacesurfer/ph2-dataset


- Paper workflow policy: configs/paper.yaml explicitly quarantines uncertain development groups and verified-copy-linked groups before splitting, without declaring pHash candidates duplicates/different. Report exclusions separately, preserve all original images/external references, fail on insufficient retained classes/groups, and never fabricate reviewer decisions. configs/default.yaml keeps strict human-review mode.
