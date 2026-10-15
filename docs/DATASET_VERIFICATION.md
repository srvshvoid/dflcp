# Dataset verification placeholder

**Status: NOT VERIFIED.** No real dataset, license record, checksum manifest,
or approved retention plan is included in this repository. The synthetic CPU
fixture must never be described as a substitute for dataset verification.

## Required record before a real run

Fill one reviewed copy of the following table for each dataset and each
derived artifact. Do not fill unknown cells with estimates.

| Field | Evidence to attach | Status |
| --- | --- | --- |
| Dataset name and release | Official release page and immutable version | Pending |
| Data controller / contact | Written authorization or institutional record | Pending |
| License and permitted use | License text, terms, and research/redistribution decision | Pending |
| Acquisition checksum | Hashes for the downloaded archive and manifest | Pending |
| Subject / source consent | Provenance and consent basis appropriate to the data | Pending |
| Sensitive-data review | Biometric/face/video handling decision and access controls | Pending |
| Preprocessing revision | Reproducible script version and dependency lock | Pending |
| Train/validation/test split | Video/source/subject disjointness evidence | Pending |
| Cross-dataset overlap | Search and exclusion process for duplicate sources | Pending |
| Label policy | Annotation guide, ambiguity handling, and adjudication record | Pending |
| Retention and deletion | Storage location, access list, retention period, deletion test | Pending |
| Release approval | Named reviewer and date for the exact artifact | Pending |

## Verification procedure

1. Obtain the dataset from an authorized source and preserve the original
   checksum without changing the archive.
2. Record license and consent evidence before extracting frames or face crops.
3. Derive a manifest containing stable source/video identifiers, labels, and
   preprocessing revision. Keep sensitive content out of the manifest unless
   its inclusion is approved.
4. Run a leakage audit at the source/video/subject level. A frame-level random
   split is insufficient for a video-forensics claim.
5. Review the split, labels, access controls, and deletion path with the data
   owner. Record the reviewer and exact artifact hash.
6. Permit a real experiment only after every required row has evidence. Keep
   the synthetic smoke replay labelled separately.

## Current gate

Until this document is filled and reviewed, FedForge may report only that its
offline packaging and synthetic replay plumbing were exercised. It may not
report real-dataset accuracy, calibration, fairness, privacy, or attack
resistance.
