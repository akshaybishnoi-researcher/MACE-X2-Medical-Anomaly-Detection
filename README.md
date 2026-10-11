# MACE-X²

## Multi-Witness Medical Image Anomaly Detection

**Research Prototype v0.1**

MACE-X² is an independent collaborative research prototype for studying reliable medical image anomaly detection using multiple evidence pathways.

The current prototype evaluates whether **global image-level evidence** and **local spatial evidence** provide complementary signals for chest X-ray anomaly detection.

> **Research use only. This project is not intended for clinical diagnosis, medical screening, or treatment decisions.**

---

## Live Research Demo

**Streamlit App:**  
https://mace-x2-research-demo.streamlit.app

The interactive demo allows users to:

- Upload a chest X-ray
- Compute a Global Witness anomaly score
- Compute a Local Witness anomaly score
- Compute the locked MACE Configuration-1 score
- View a 7 × 7 local spatial evidence map
- Inspect validation-locked research thresholds

---

## Researchers

**Akshay Bishnoi**  
**Anshu Madheshiya**

Independent Collaborative Research Project

---

## Research Objective

Medical anomaly detection systems can appear reliable when multiple model components produce similar outputs.

However, apparently different signals may actually contain highly redundant information.

MACE-X² explores a framework where evidence pathways are evaluated not only by predictive performance, but also by:

- Complementarity
- Redundancy
- Stability under intervention
- Local and global consistency
- Future counterfactual and abstention mechanisms

The long-term objective is to study whether anomaly detection systems can produce more reliable evidence by combining multiple independently evaluated witnesses.

---

## Dataset

This prototype uses the **NIH ChestX-ray14** dataset.

### Primary experimental cohort

- PA-view chest radiographs only
- Patient-disjoint train, validation, and test partitions
- Training reference contains conservative-normal patients only
- No abnormal-labelled images are used in the normal training reference

### Strict primary cohort

| Split | Images |
|---|---:|
| Train | 13,870 |
| Validation | 16,794 |
| Test | 17,119 |

Training images are derived only from conservative-normal patients.

Validation and test positives contain abnormal-labelled images.

---

## Leakage-Safe Experimental Design

Patient-level separation was used to prevent images belonging to the same patient from appearing across multiple splits.

The initial dataset contained:

- **112,120 images**
- **30,805 patients**

The primary experiment was restricted to PA-view images after an acquisition-view audit identified substantial AP/PA imbalance.

---

# Baselines

## Baseline 1 — Frozen ResNet18 + Cosine kNN

Pipeline:

1. Resize chest X-ray to 224 × 224
2. Convert grayscale image to three channels
3. Apply ImageNet normalization
4. Extract a 512-dimensional embedding using frozen ImageNet-pretrained ResNet18
5. Compare the embedding against conservative-normal training references
6. Use cosine 3-nearest-normal distance as the anomaly score

### Test performance

- **AUROC:** 0.6624
- **AUPRC:** 0.8943

---

## Baseline 2 — PCA Reconstruction Anomaly

A second baseline family was created using PCA reconstruction error on the frozen ResNet18 embeddings.

### Test performance

- **AUROC:** 0.6253
- **AUPRC:** 0.8816

Paired patient-cluster bootstrap comparison:

**kNN − PCA AUROC difference:**  
`+0.0371`

95% CI:

`[+0.0281, +0.0459]`

The cosine kNN baseline consistently outperformed the PCA baseline under this evaluation protocol.

---

# MACE Configuration-1

The first MACE prototype combines two evidence pathways derived from the same frozen ResNet18 backbone.

## Global Witness

- Whole-image 512-dimensional representation
- Cosine 3-nearest-normal anomaly score

## Local Witness

- ResNet18 layer4 spatial representation
- 7 × 7 feature grid
- 49 position-aware local feature vectors
- One conservative-normal prototype per spatial position
- Mean anomaly evidence across all 49 locations

The 7 × 7 locations represent **CNN spatial feature positions**, not clinically segmented anatomical regions.

---

## Locked Evidence Combination

Validation-selected weights:

- **Global Witness:** 0.75
- **Local Witness:** 0.25

### Validation

- Global-only AUROC: **0.6560**
- Local-only AUROC: **0.6377**
- MACE C1 AUROC: **0.6571**

### Locked test evaluation

| Method | Test AUROC | Test AUPRC |
|---|---:|---:|
| Frozen ResNet18 + cosine 3-NN | **0.6624** | **0.8943** |
| PCA reconstruction | 0.6253 | 0.8816 |
| MACE Configuration-1 | 0.6606 | 0.8931 |

MACE Configuration-1 did **not** demonstrate a clear improvement over the strongest global-only baseline.

---

# Witness Redundancy

The Global and Local witnesses showed substantial statistical dependence:

- **Pearson correlation:** 0.8107
- **Spearman correlation:** 0.7911

This finding is important because it demonstrates that combining multiple evidence pathways does not automatically create independent evidence.

The result motivates future redundancy-aware MACE-X² mechanisms.

---

# Paired Patient-Cluster Bootstrap

MACE C1 vs Global-only:

### AUROC difference

`-0.0018`

95% CI:

`[-0.0043, +0.0007]`

The confidence interval includes zero.

Therefore, no clear AUROC improvement was demonstrated.

### AUPRC difference

`-0.0012`

95% CI:

`[-0.0021, -0.0002]`

This represents a small but consistently negative AUPRC difference under the bootstrap procedure.

---

# Intervention Robustness Audit

A validation-only acquisition-like intensity intervention experiment was performed using fixed brightness transformations:

- 0.90 × brightness
- 1.10 × brightness

No intervention strength was tuned after observing the results.

| Method | Mean absolute AUROC change | Mean Spearman stability |
|---|---:|---:|
| Global | 0.0044 | 0.9816 |
| MACE C1 | 0.0043 | 0.9853 |

MACE C1 showed a modest ranking-stability advantage under these perturbations, although the difference was small.

This experiment is interpreted as a controlled robustness audit, **not causal proof**.

---

# Interactive Demo

The Streamlit application reproduces the locked MACE Configuration-1 inference pipeline.

For an uploaded chest X-ray, the interface displays:

### Global Witness

Cosine distance from the normal training embedding reference bank.

### Local Witness

Average position-aware spatial anomaly score across the 7 × 7 layer4 representation.

### MACE C1 Score

Validation-standardized weighted combination of the Global and Local witnesses.

### Local Spatial Evidence Map

A 7 × 7 visualization of CNN spatial anomaly evidence.

The map should not be interpreted as lesion localization or anatomical segmentation.

---

# Repository Structure

```text
MACE-X2-Medical-Anomaly-Detection/
│
├── app.py
│
├── src/
│   └── inference.py
│
├── artifacts/
│   ├── features.npy
│   ├── local_normal_position_prototypes.npy
│   ├── mace_configuration1_locked_protocol.json
│   └── mace_configuration1_locked_thresholds.json
│
├── results/
├── notebooks/
├── docs/
├── assets/
│
├── requirements.txt
├── .gitignore
└── README.md

git clone https://github.com/akshaybishnoi-researcher/MACE-X2-Medical-Anomaly-Detection.git

cd MACE-X2-Medical-Anomaly-Detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py