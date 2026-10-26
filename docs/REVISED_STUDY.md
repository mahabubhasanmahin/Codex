# Current approved study: PAD + HAM development, PH2 external

The latest user-approved design supersedes the original PDF's three-source
proposal and the interim PAD+Dermnet/HAM+PH2 protocol. The title remains
CrossSkinNet: Explainable Multimodal Skin Disease Classification Under Missing
Metadata. The dataset-role change must be described in the manuscript.

|Source|Current role|Enabled by default|Grouping|
|---|---|---|---|
|PAD-UFES-20|Development|True|patient_id|
|HAM10000|Development|True|lesion_id|
|PH2|External test/reference only|Always True|Image identifier; no inspected patient/lesion IDs|
|Dermnet|Optional reviewed image-only development|False|Image fallback; patient independence unverified|
|ISIC2018|Excluded|False|Not used in current experiments|

Both PAD and HAM must survive into the actual training split. Each contributes
images and observed age/sex/site, with symptoms available only in PAD. Metadata
is never statistically imputed. HAM lesion grouping is not proof of patient-level
independence across different lesions when patient IDs are unavailable.

## Leakage and evaluation rules

1. PH2 images never enter training, validation, class weighting, early stopping or
   temperature calibration. All raw PH2 images remain overlap references, even if
   a PH2 diagnosis is excluded from the final evaluation taxonomy.
2. Remove SHA-exact or human-confirmed PH2 duplicates from development. Generate
   pHash candidates at the configured radius but do not auto-remove hash collisions.
   Pending relevant reviews block final dataset construction and training.
3. Within PAD+HAM development, verified duplicate components keep the best-metadata
   training-pool row before splitting. A PAD-HAM duplicate is an INTERNAL duplicate
   in this protocol, not an external-overlap removal.
4. Group by recorded PAD patient IDs and HAM lesion IDs before approximate 70/15/15
   splitting. The source schemas explicitly require those fields; missing required
   identifiers fail rather than falling back to image-level groups.
5. The pooled PAD+HAM model has in-domain test results for both sources. Do not call
   its HAM test results zero-shot. PH2 is the only external source by default,
   limited to compatible reviewed disease labels. Its inspected data has no model
   metadata, so its predictions use the calibrated image-only fallback.
6. PAD-only→HAM and HAM-only→PAD experiments are separate study protocols, not
   results from the pooled run. They require separate source roles, vocabulary,
   overlap checks and checkpoints; the current defaults do not implement those
   training experiments automatically.

## Current shared-three-class mapping

The selected operational codebook now maps canonical PAD/HAM MEL/BCC/NEV codes
to melanoma, basal_cell_carcinoma and melanocytic_nevus. See THREE_CLASS_PROTOCOL.md
for exact spellings, exclusions and source-verification limits. This is not new
clinical reannotation or independent dermatologist validation. SCC is not AK/AKIEC; PAD
SEK is not automatically the entire HAM BKL category. PH2 common/atypical nevus
mapping must be explicit. No clinical or anatomical mapping was auto-approved.
Dermnet remains disabled; enabling it requires reliable disease-level labels and
acknowledging that image-ID grouping does not establish patient independence.

## Reproduce the current setup on Kaggle

Attach PAD-UFES-20, HAM10000 and PH2. ISIC and Dermnet are not required.

```python
import subprocess, sys
subprocess.run(['git','pull','--ff-only','origin','main'], check=True)
subprocess.run([sys.executable,'scripts/00_audit.py',
                '--config','configs/default.yaml',
                '--schemas','configs/dataset_schemas.yaml'], check=True)
```

The audit merges the actual attached data and writes provisional counts and
metadata/label-review reports. source_selection.csv records PAD/HAM development,
PH2 external and the two disabled sources. Then follow stages 02–07 after medical
label/site review and relevant duplicate decisions. Raw ISIC or historical
HAM-external counts must not be reused as current-protocol results.

Use a fresh output_root if retaining historical artifacts. Old tables with HAM in
raw_test_pool.csv are rejected under the current configuration. Previously built
models are not converted by changing config: regenerate pools/splits and train new
checkpoints. Final dataset README/statistics list the actual retained sources.

Historical fixtures explicitly configure legacy roles for regression testing.
They never change the current default roles or imply that historical experiments
were performed on real medical data.
