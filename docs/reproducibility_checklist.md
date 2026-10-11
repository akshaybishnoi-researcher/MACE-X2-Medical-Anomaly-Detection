# Reproducibility Checklist

This checklist documents the main reproducibility safeguards used in the MACE-X² Research Prototype v0.1.

## Dataset and Cohort

- [x] NIH ChestX-ray14 dataset provenance documented
- [x] Dataset audited before modelling
- [x] Patient identifiers used for split construction
- [x] Primary experiment restricted to PA-view chest radiographs
- [x] Conservative-normal training reference defined explicitly
- [x] No abnormal-labelled images included in the normal training reference

## Leakage Control

- [x] Train, validation, and test splits are patient-disjoint
- [x] No patient appears in more than one split
- [x] Model-selection decisions use validation data rather than test data

## Baselines

- [x] Frozen ResNet18 + cosine kNN baseline evaluated
- [x] PCA reconstruction baseline evaluated
- [x] Baselines use the same primary cohort
- [x] Test comparisons reported without hiding non-improving results

## MACE Configuration-1

- [x] Global Witness specification documented
- [x] Local Witness specification documented
- [x] Local spatial locations are not claimed to be anatomical segmentations
- [x] Global and Local fusion weights selected using validation data
- [x] Thresholds locked before final MACE test evaluation

## Statistical Evaluation

- [x] Test AUROC and AUPRC reported
- [x] Patient-cluster bootstrap used
- [x] Paired comparison used when comparing methods
- [x] Confidence intervals reported for method differences
- [x] Witness redundancy measured using Pearson and Spearman correlation

## Robustness

- [x] Fixed brightness interventions evaluated
- [x] Intervention strengths were not tuned after observing results
- [x] Robustness experiment treated as an acquisition-like perturbation audit
- [x] No causal-independence claim made from the perturbation experiment

## Limitations and Safety

- [x] NIH label noise acknowledged
- [x] `No Finding` is not treated as confirmed clinical health
- [x] No clinical diagnostic claim is made
- [x] Spatial evidence map is described as CNN feature evidence, not lesion localization
- [x] MACE Configuration-1 non-improvement over the strongest baseline is reported explicitly

## Reproducibility Status

The current v0.1 release provides a reproducible proof-of-concept including:

- locked experimental protocol,
- baseline comparisons,
- multi-witness inference,
- redundancy analysis,
- patient-cluster bootstrap evaluation,
- intervention robustness audit,
- and an interactive Streamlit research demo.

This checklist was reviewed as part of the collaborative project documentation.
