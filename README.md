# SCARF-SSL-Microarray-Leukemia

A Self-Supervised Contrastive Learning (SCARF) framework for multi-class leukaemia subtype classification using high-dimensional microarray gene expression data.

---

## Overview

This repository contains the full benchmarking pipeline behind a Self-Supervised Contrastive Learning (SCARF) framework for leukaemia subtype classification using the MILE (Microarray Innovations in Leukaemia) study dataset. It matches the manuscript's Methods section end to end.

The framework includes:

- SCARF-based self-supervised contrastive pre-training, plus four biologically-motivated tabular augmentations (Gaussian noise, gene masking, probe subsetting, magnitude warping)
- Two-stage supervised fine-tuning (linear probing, then end-to-end with a learning-rate scheduler)
- Baseline comparators: Random Forest, SVM-RBF, XGBoost, and a from-scratch supervised MLP
- Stratified 5-fold cross-validation, with paired t-test and Wilcoxon significance testing
- A label-efficiency sweep across 5%, 10%, 20%, and 100% labelled data
- SHAP interpretability via a PCA + logistic-regression surrogate, back-projected to gene-level importance
- Pathway enrichment (Enrichr / GO / KEGG) on the top-ranked genes
- External cohort validation on an independent dataset (GSE9476), with probe alignment across microarray platforms

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
│   ├── run_pipeline.py
│   ├── requirements.txt
│   └── scarf_ssl/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── augmentations.py
│       ├── models.py
│       ├── losses.py
│       ├── training.py
│       ├── baselines.py
│       ├── stats.py
│       ├── evaluation.py
│       ├── interpretability.py
│       └── external_validation.py
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

Run the full pipeline:

```bash
python code/run_pipeline.py
```

Because the full pipeline runs ~30 complete training passes (5-fold CV × [Table 2 + 4 label-efficiency fractions] + 5 augmentation-ablation runs + 1 final model), it's expensive to run in full. For a quick first check that everything works end to end, skip the slower stages:

```bash
python code/run_pipeline.py --skip-label-efficiency --skip-ablation --skip-shap --skip-enrichment --skip-external
```

Run `python code/run_pipeline.py --help` for the full list of options (epochs, batch size, results directory, etc.).

**Two steps require manual input before they'll produce meaningful results:**

1. **Pathway enrichment** needs gene *symbols*, not the raw probe IDs the pipeline outputs by default — map probes to HGNC symbols using GPL7473's platform annotation table first (see `scarf_ssl/interpretability.py`).
2. **External validation** (GSE9476) needs a manual mapping from the 18 MILE-cohort classes to GSE9476's own class taxonomy — see `scarf_ssl/external_validation.py` for how to supply this.

---

## Outputs

Saved to the `results/` directory:

- `table2_cv_raw.csv` / `table2_cv_summary.csv` — 5-fold CV benchmark across all methods
- `table3_label_efficiency_raw.csv` / `_summary.csv` — performance across labelled-data fractions
- `table4_augmentation_ablation.csv` — per-augmentation performance
- `significance_tests.csv` — paired t-test and Wilcoxon results
- `final_model.pt` and `scaler.pkl` — the trained SSL model and its fitted scaler
- `shap_top_genes.csv` / `shap_top_genes.png` — SHAP-based gene importance
- `pathway_enrichment_significant.csv` / `pathway_enrichment_top10.png` — enriched pathways (once gene-symbol mapping is supplied)

---

## Citation

If you use this repository in your research, please cite the associated publication (when available) and this software repository.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
