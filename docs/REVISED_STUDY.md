# User-approved study revision: PAD + reviewed Dermnet

The original PDF proposed PAD-UFES-20 + ISIC + Dermnet for development and
HAM10000/PH2 for external testing. The user approved excluding the currently
attached overlapping ISIC2018 copy. This is a documented data-design revision;
it is not an assertion that the original three-source experiment was completed.

## Active source configuration

|Source|Role|Default enabled|
|---|---|---|
|PAD-UFES-20|Development, multimodal|True|
|Dermnet|Development, reviewed disease labels only|True|
|Current ISIC2018 copy|Excluded; overlaps external references|False|
|HAM10000|External test/reference only|Always True|
|PH2|External test/reference only|Always True|

ISIC2018's path/schema remain available for reproducing historical audits, not
for the revised default training run. The disabled source is never opened by
merge; it need not remain attached on Kaggle. HAM/PH2 cannot be disabled or moved
into development. All raw external images protect against overlap, even if their
labels are later excluded from evaluation.

## What changes and what stays

- The architecture, four-backbone comparison, missingness experiments,
  validation-only calibration, explanations and held-out/external testing stay.
- The dataset plan changes from three proposed sources to two configured sources.
  Methods/results must report actual retained sources and counts after filtering.
- Dermnet ingestion still reads all configured folders for transparent inventory.
  It does not approve mixed/broad folder names as specific diseases. Label mapping
  is an independent review step and remains unfilled where decisions are unknown.
- No clinical mappings or anatomical site mappings were automatically invented.
  The default study requires both PAD and Dermnet in the actual training split;
  build/train fail if one disappears after label selection or deduplication.
  A future training run can only claim PAD+Dermnet if reviewed labels from both
  actually survive. The generated README/statistics name actual retained sources.
- Dermnet lacks inspected patient/lesion identifiers. Its image-ID fallback does
  not establish patient independence; the limitation must appear in the paper.
- Source-dependent metadata availability can confound classification. Compare
  image-only and multimodal baselines and report per-source results explicitly.
- Old provisional counts and pending-review totals belong to the earlier input
  pool. Rerun merge/audit and downstream stages; do not reuse old derived artifacts
  or checkpoints as revised-study results.

## Kaggle commands

From the repository root:

```python
import subprocess, sys
subprocess.run(['git','pull','--ff-only','origin','main'], check=True)
subprocess.run([sys.executable,'scripts/00_audit.py',
                '--config','configs/default.yaml',
                '--schemas','configs/dataset_schemas.yaml'], check=True)
```

The audit refreshes merge and writes only provisional data until duplicate review
is complete. Inspect source_selection.csv: PAD/Dermnet are enabled, ISIC2018 is
excluded with its reason, and HAM/PH2 remain external. Then use stages 02–07 from
RUN_ON_KAGGLE.md after disease/site mapping and relevant duplicate decisions.

Existing unused ISIC mapping rows can remain in the editable CSV. They do not
need invented values when their labels are absent from both active pools. Stale
raw/labeled/normalized/unified training tables containing ISIC are rejected under
the revised configuration. Use a new output_root for new experiments if you need
to retain a previous experiment; no checkpoint/run directory is silently replaced.
