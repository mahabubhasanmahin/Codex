# Upload the prepared data privately

1. Run stages 01–05 on Kaggle after reviewing label and site mappings.
2. Download or retain the complete `CrossSkinNet_Unified_Dataset` folder beneath
   your configured output root, including both CSVs, images and audit statistics.
3. In Kaggle, choose **Create → New Dataset**, upload that folder, and select
   **Private** visibility. Check each source license before sharing images.
4. Attach the new private dataset to a training notebook. Set `paths.unified_root`
   in a copy of `configs/default.yaml` to the mounted folder containing metadata.csv.
   Keep `paths.output_root` pointed to writable working storage for checkpoints/results.
5. Confirm the `split` and `group_id` columns before training. Do not rebuild a
   different split per backbone, and do not use zero_shot.csv for training/calibration.
