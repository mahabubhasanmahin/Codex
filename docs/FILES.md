# File inventory

| File | Purpose |
|---|---|
| `.gitignore` | Ignore Python caches and local environments. |
| `AGENTS.md` | Coding, data-safety and per-step verification rules. |
| `README.md` | Project entry point, demo instructions and validation limits. |
| `configs/dataset_schemas.yaml` | Inspection-derived loader layout, column and label configuration. |
| `configs/default.yaml` | Dataset paths, pool membership, seeds and experiment hyperparameters. |
| `configs/label_mapping.csv` | Editable disease taxonomy decisions; unknown mappings remain blank. |
| `configs/site_mapping.yaml` | Editable anatomical mappings; inventory fills unreviewed raw values. |
| `docs/FILES.md` | One-line inventory of implementation files. |
| `docs/IMPLEMENTATION.md` | Paper alignment, scientific choices and unresolved data limitations. |
| `docs/RUN_ON_KAGGLE.md` | Ordered notebook cells for synthetic and real-data workflows. |
| `docs/upload_to_kaggle.md` | Private hosting of the derived unified dataset. |
| `notebooks/CrossSkinNet_Kaggle.ipynb` | Beginner-facing Kaggle demo notebook. |
| `requirements.txt` | Runtime and test dependencies. |
| `scripts/00_smoke.py` | Run the entire pipeline with generated fake data and no weight downloads. |
| `scripts/01_merge.py` | Thin CLI for schema-driven raw pool ingestion. |
| `scripts/02_labels.py` | Thin CLI for label inventory and reviewed mapping application. |
| `scripts/03_metadata.py` | CLI for site inventory and missing-metadata normalization. |
| `scripts/04_dedup.py` | CLI for duplicate hashing, removal and manual-audit outputs. |
| `scripts/05_build_unified.py` | CLI for group splits and the portable unified dataset. |
| `scripts/06_train.py` | CLI for reproducible CrossSkinNet training and validation calibration. |
| `scripts/07_evaluate.py` | CLI for held-out, zero-shot, robustness and explainability evaluation. |
| `src/crossskinnet/__init__.py` | CrossSkinNet package. |
| `src/crossskinnet/data/__init__.py` | CrossSkinNet data. |
| `src/crossskinnet/data/build_unified.py` | Group-stratified development splits and a portable image dataset artifact. |
| `src/crossskinnet/data/dedup.py` | Cached exact/perceptual duplicate components using a 64-bit Hamming BK-tree. |
| `src/crossskinnet/data/labels.py` | Explicit, editable label harmonization; no medical equivalences are inferred. |
| `src/crossskinnet/data/merge.py` | Schema-driven raw dataset ingestion with strictly separate pool outputs. |
| `src/crossskinnet/data/metadata.py` | Explicit metadata normalization, structural missingness and field-level masking. |
| `src/crossskinnet/data/synthetic.py` | Generate an explicitly synthetic layout; this does not describe real datasets. |
| `src/crossskinnet/demo.py` | One-command synthetic pipeline demonstration, never a thesis experiment. |
| `src/crossskinnet/eval/__init__.py` | CrossSkinNet eval. |
| `src/crossskinnet/eval/evaluate.py` | In-domain, zero-shot, missingness, explainability and backbone result artifacts. |
| `src/crossskinnet/eval/explain.py` | Grad-CAM image explanations and field-coalition SHAP metadata attributions. |
| `src/crossskinnet/eval/metrics.py` | Classification, calibration and group-bootstrap uncertainty metrics. |
| `src/crossskinnet/io.py` | Configuration, CSV persistence and reproducible reporting helpers. |
| `src/crossskinnet/models/__init__.py` | CrossSkinNet models. |
| `src/crossskinnet/models/backbones.py` | Pooled and token embeddings from tiny, timm and Hugging Face image encoders. |
| `src/crossskinnet/models/network.py` | CrossSkinNet: learned missing tokens, gated cross-attention and image fallback. |
| `src/crossskinnet/train/__init__.py` | CrossSkinNet train. |
| `src/crossskinnet/train/dataset.py` | Dataset and preprocessing shared by training, calibration and evaluation. |
| `src/crossskinnet/train/engine.py` | Deterministic training, resumable epoch checkpoints and validation-only calibration. |
| `tests/__init__.py` | Offline synthetic test support. |
| `tests/conftest.py` | Offline prepared pipeline fixture shared by training and evaluation tests. |
| `tests/fixtures/__init__.py` | Offline synthetic test support. |
| `tests/fixtures/make_synthetic_data.py` | Public test fixture entry point for the shared offline synthetic generator. |
| `tests/test_build_unified.py` | Disjoint group splitting and portable image export. |
| `tests/test_dedup.py` | Duplicate priority, metadata preference and cache repeatability. |
| `tests/test_evaluation.py` | Offline end-to-end held-out, robustness, SHAP and Grad-CAM validation. |
| `tests/test_inspected_layouts.py` | Offline miniature datasets that exercise the inspected production schemas. |
| `tests/test_labels.py` | Mapping completeness, explicit exclusions and zero-shot class eligibility. |
| `tests/test_merge.py` | Exercise real ingestion code against explicitly configured synthetic data. |
| `tests/test_metadata.py` | Field normalization and reproducible mask protocols. |
| `tests/test_model_training.py` | Offline tiny-backbone routing, learnability and CLI training smoke test. |
| `tests/test_synthetic_data.py` | Validate the offline fixture's leakage cases and missing metadata. |
| `scripts/00_audit.py` | Mapping-free dataset feasibility audit CLI. |
| `src/crossskinnet/data/audit.py` | Source/label, metadata and post-duplicate image/group reports. |
| `tests/test_audit.py` | Offline audit CLI, repeatability and pipeline-isolation checks. |
| `src/crossskinnet/data/grouping.py` | Explicit patient/global lesion scope and validated patient grouping. |
| `scripts/00_review_overlap.py` | Exact/perceptual evidence and focused overlap review CLI. |
| `docs/AUDIT_FINDINGS.md` | Findings and unresolved limitations from the uploaded real-data reports. |
| `src/crossskinnet/data/sources.py` | Source enable flags, fixed external roles and stale-pool validation. |
| `tests/test_source_selection.py` | Revised default, absent disabled input and active-only label validation tests. |
| `docs/REVISED_STUDY.md` | Approved two-source revision, scientific limits and Kaggle steps. |
| `src/crossskinnet/data/review.py` | Selected-label candidate preparation and atomic per-pair decision persistence. |
| `src/crossskinnet/data/review_ui.py` | Notebook image-pair panel with explicit decision buttons. |
| `scripts/04_prepare_review.py` | Mapping and selected-label duplicate-review preview CLI. |
| `tests/test_selected_review.py` | Canonical mapping, external-reference protection and saved-widget-decision checks. |
| `docs/THREE_CLASS_PROTOCOL.md` | Exact selected taxonomy, exclusions and source-verification limits. |
| `notebooks/02_three_class_review.ipynb` | Kaggle bootstrap and interactive duplicate-review workflow. |
