# DIDA

**DIDA** (Data-Importance driven Differential Augmentation) is a value-driven framework for tabular data augmentation in low-data, high-constraint scenarios (e.g., medical diagnosis, financial risk control).

## Overview

DIDA augments tabular data by assigning a *perturbation budget* to each sample-feature cell according to its data value, which is jointly encoded from:

- **Feature-level** SHAP importance, and
- **Sample-level** Shapley (Data Shapley) contribution.

The optimal perturbation strength is inversely related to data value — high-value cells receive small perturbations, low-value cells receive larger ones — giving a "high-value micro-perturb, low-value macro-perturb" strategy. DIDA additionally builds a constraint table (ANOVA–MI fused scoring) so that categorical feature replacement respects domain hard constraints, and it records an auditable augmentation log for traceability.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python test_epfinal_5.py
```

The script loads `heart.csv`, trains a baseline, computes feature/sample importance, constructs the complete domain and constraint table, performs differential augmentation, and re-evaluates the augmented data across multiple classifiers.

## Repository structure

```
.
├── test_epfinal_5.py   # DIDA pipeline (data loading → importance → augmentation → evaluation)
├── heart.csv           # demo dataset
├── pytorch_tabnet/     # self-contained TabNet classifier (paper-faithful implementation)
│   └── tab_model.py
├── requirements.txt
└── README.md
```

## Notes

- The demo dataset is `heart.csv`. Other evaluation datasets used in the paper are not included in this repository.
- All generated outputs (figures and intermediate CSVs) are written to the working directory and are git-ignored.
