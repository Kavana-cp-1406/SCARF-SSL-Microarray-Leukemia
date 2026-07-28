# SCARF-SSL-Microarray-Leukemia

A Self-Supervised Contrastive Learning (SCARF) framework for multi-class leukemia subtype classification using high-dimensional microarray gene expression data.

---

## Overview

This repository contains the implementation of a Self-Supervised Contrastive Learning (SCARF) framework for leukemia subtype classification using the MILE (Microarray Innovations in Leukemia) study dataset.

The framework includes:

- Self-supervised contrastive representation learning
- Multi-class leukemia subtype classification
- SHAP-based model interpretation
- t-SNE visualization
- Stratified cross-validation
- External validation

---

## Dataset

- **Dataset:** GSE13164
- **Platform:** GPL7473
- **Source:** NCBI Gene Expression Omnibus (GEO)
- **Samples:** 1,152
- **Diagnostic classes:** 18

The dataset is automatically downloaded using GEOparse. Raw microarray data are not redistributed in this repository.

---

## Repository Structure

```text
SCARF-SSL-Microarray-Leukemia/
├── code/
│   └── scarf_ssl_leukemia.py
├── data/
│   └── README.md
├── results/
├── figures/
├── notebooks/
├── checkpoints/
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kavana-cp-1406/SCARF-SSL-Microarray-Leukemia.git
cd SCARF-SSL-Microarray-Leukemia
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the pipeline:

```bash
python code/scarf_ssl_leukemia.py
```

---

## Outputs

The framework generates:

- Classification performance metrics
- Trained SCARF model
- SHAP feature importance analysis
- t-SNE visualizations
- Saved model checkpoints

---

## Citation

If you use this repository in your research, please cite the associated publication (when available) and this software repository.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
