# Findings from the uploaded real-data audit

> Correction after reviewing the targeted grids: the old pHash-only removal policy
> was too aggressive. The removed counts below describe that historical run, not
> confirmed duplicate counts or a training-ready dataset. Current code removes
> only SHA-exact or explicitly reviewed duplicates. All other pHash pairs (even
> distance zero) remain candidates. Rerun the audit for provisional verified-only
> counts, review candidates, then rerun stage 04 before final counts/training.

Evidence: user-supplied audit_reports.zip, raw_train_pool.csv and
duplicates_report.csv. Images themselves were not present on this cloud machine;
the duplicate grid supplied in chat is illustrative, not a review of every pair.
No diagnoses are inferred from those images.

## ISIC2018 / HAM10000

- Raw ISIC2018: 10,015 images; audit retains zero after external-overlap removal.
- SHA256-equality components in the reported edges connect 9,998 ISIC training
  records to external images. This is a proven lower bound: the report stores
  representative edges and can omit additional equality pairs in a pHash bucket.
- Equal-pHash components connect all 10,015 ISIC images to external data. The
  remaining 17 are not proven byte-equal from this report alone.
- Do not count this ISIC2018 copy as independent training data while HAM10000
  remains external. Keeping it requires violating the stated leakage rules.
  A different ISIC subset must be independently inspected/deduplicated before
  claiming the original three-source training plan was actually realized.

## PAD identifiers

- Raw PAD: 2,298 rows, 1,373 recorded patients, 1,641 bare lesion numbers and
  1,891 distinct (patient_id, lesion_id) pairs.
- 250 bare lesion values occur under multiple patient IDs in raw data (247 in
  the retained-data audit). This alone does not prove erroneous patient IDs.
- All 2,298 image IDs begin with the exact configured patient_id + '_' +
  lesion_id + '_' prefix. The schema now declares patient-scoped lesion IDs
  and verifies that prefix during grouping. Patient IDs remain the split unit.
- The earlier build guard assumed global lesion IDs and would incorrectly
  reject these data. Global/default scopes still reject ambiguous associations;
  missing patient IDs for declared patient-scoped lesions fail explicitly.
- Counted identifiers and filename agreement are not independent verification
  of biological identity. Patient splits still require dataset documentation review.

## Perceptual overlap candidates requiring targeted review

|Training source|Connected through reported SHA-only edges|Equal-pHash edges|All reported pHash edges|
|---|---:|---:|---:|
|ISIC2018|9998|10015|10015|
|PAD-UFES-20|0|0|13|
|Dermnet|0|0|17|

The 13 PAD and 17 Dermnet removals are connected to external components through
positive-distance perceptual edges. No direct PAD–external edge appears in this
report; its removals are transitive through other training nodes. These candidates
must not be described as proven exact duplicates. Do not automatically restore
or permanently accept their removal without inspecting the relevant components.

Run 00_review_overlap.py after pulling the update. It reuses existing reports,
writes overlap_evidence.csv and exports targeted PAD/Dermnet edge CSVs and sampled
grids. The CSVs contain all relevant reported edges; each grid samples up to 20.
The dataset paths in those reports must remain available on the Kaggle machine.

## Still unresolved

Mixed Dermnet diagnosis folders cannot be renamed to one disease. Folder-based
labels are not automatically invalid either: specific candidates such as Atopic
Dermatitis Photos and Urticaria Hives need source-label verification before use.
No disease taxonomy was silently approved, no source was replaced and no real
training was launched as part of this investigation.
