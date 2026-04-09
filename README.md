# Cycloplegic Refraction Prediction: Eyerobo VS vs Auto-Refractor

Predicting cycloplegic refraction from non-cycloplegic measurements using machine learning in pediatric myopia. This repository provides the dataset (1,216 eyes from 609 patients, ages 6–19), experiment code, and figure generation scripts for reproducing the results reported in our paper.

The study compares the Eyerobo Vision Screener (a portable handheld photorefractor) against a conventional auto-refractor (AR-1, Nidek, Japan) across 12 feature scenarios, predicting cycloplegic spherical equivalent (SE) and power vector components (J0, J45) using TabPFN and 7 other ML models.

![Pipeline](figures/arch.png)

## Key results

| Configuration | MAE (D) | Within ±0.50 D |
|--------------|---------|----------------|
| Auto-refractor Pre (TabPFN) | 0.497 | 76.2% |
| Eyerobo Pre (TabPFN) | 0.786 | 59.6% |
| Eyerobo + IOL Pre (TabPFN) | 0.616 | 62.7% |
| Auto-refractor Post (TabPFN) | 0.220 | 95.4% |

TabPFN achieved the lowest MAE on all 12 scenarios. IOL Master biometry reduced the Eyerobo–auto-refractor gap by 59%. For mild myopia screening, the Eyerobo achieved MAE = 0.383 D (77.7% within ±0.50 D).

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended for TabPFN)

```bash
pip install numpy pandas scikit-learn scipy matplotlib
pip install xgboost lightgbm catboost
pip install tabpfn
```

## Dataset

The dataset is in `data/dataset.csv` — 1,216 eyes from 609 pediatric patients (ages 6–19). All patient identifiers have been anonymized.

**Columns:**
- `patient_id` — anonymous patient identifier (P0001, P0002, ...)
- `eye_position` — OD (right) or OS (left)
- `gender`, `age`, `IOP`, `treatment` — demographics
- `target_SPH`, `target_CYL`, `target_AX`, `target_SE` — cycloplegic refraction (gold standard)
- `eyerobo_pre_*`, `eyerobo_post_*` — Eyerobo VS measurements (pre/post dilation)
- `auto_pre_*`, `auto_post_*` — auto-refractor measurements (pre/post dilation)
- `iol_pre_*`, `iol_post_*` — IOL Master biometry (pre/post dilation)

## Repository structure

```
├── data/
│   └── dataset.csv              # Anonymized dataset (1,216 eyes)
├── experiments/code/
│   ├── prepare.py               # Data loading, feature groups, metrics
│   ├── clean_and_analyze_data.py # Data cleaning and descriptive statistics
│   ├── run_comprehensive_experiments.py  # 12 scenarios × 8 models (SE)
│   ├── run_power_vector_experiments.py   # SE + J0 + J45 predictions
│   └── ablations.py             # Dilation/IOL/learning curve ablations
├── scripts/
│   └── generate_figures.py      # Publication figure generation
└── figures/                     # Pre-generated publication figures
```

## Running experiments

### Main experiment (12 scenarios × 8 models)

```bash
cd experiments/code
python run_comprehensive_experiments.py
```

This runs all 12 feature scenarios with 8 ML models (TabPFN, XGBoost, LightGBM, CatBoost, Random Forest, Ridge, SVR, MLP) using 5-fold patient-level GroupKFold cross-validation repeated over 3 seeds. Results are saved to `experiments/comprehensive_results.json`. Runtime: ~15 minutes on GPU.

### Power vector experiments (SE + J0 + J45)

```bash
cd experiments/code
python run_power_vector_experiments.py
```

Predicts all three power vector components across 12 scenarios with 4 models. Results saved to `experiments/power_vector_results.json`. Runtime: ~25 minutes on GPU.

### Ablation studies

```bash
cd experiments/code
python ablations.py
```

Runs device leave-one-out, engineered SE feature, learning curve, and error distribution ablations. Results saved to `experiments/ablation_results.json`.

### Generate figures

```bash
cd scripts
python generate_figures.py
```

Generates all publication figures (demographics, Bland-Altman plots, scatter plots, learning curve) as PNG + PDF in `figures/`.

## 12-scenario framework

| # | Scenario | Features | Total |
|---|----------|----------|-------|
| S01 | AR Pre | SPH, CYL, AX + demographics | 8 |
| S02 | AR Post | SPH, CYL, AX + demographics | 8 |
| S03 | AR Pre+Post | SPH, CYL, AX (both) + demographics | 11 |
| S04 | ER Pre | SPH, CYL, AX + demographics | 8 |
| S05 | ER Post | SPH, CYL, AX + demographics | 8 |
| S06 | ER Pre+Post | SPH, CYL, AX (both) + demographics | 11 |
| S07–S12 | Same as S01–S06 + IOL Master biometry (9 features) | 17–20 |

## License

MIT License. See [LICENSE](LICENSE) for details.
