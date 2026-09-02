# XGBoost–SHAP Explainability Framework

A reproducible pipeline for benchmarking XGBoost against simpler classifiers on student
dropout/intervention prediction, paired with SHAP explainability, applied across three
related analyses:

- **Benchmark** — dropout prediction on a public institutional dataset.
- **Ghana institutional** — a course-level early-warning model built from per-course grade
  records, predicting which students need intervention (`OverallScore < 50`). The final
  exam score and overall score are excluded from the model inputs to avoid target leakage.
- **Lecturer survey** — descriptive statistics and reliability (Cronbach's alpha) for a
  10-construct Human-AI trust / SHAP explanation clarity survey of lecturers.

For all three, four models are compared — Logistic Regression, Decision Tree, Random Forest,
and XGBoost — with SHAP (global bar, beeswarm, and local waterfall plots) used to explain the
XGBoost model.

## Repository layout

```
code/
  fast_canonical_rebuild.py            # deterministic rebuild of all figures/results (no CV)
  full_rebuild_with_optional_10fold_cv.py  # same pipeline plus a 10-fold CV summary
notebooks/
  Updated_XGBoost_SHAP.ipynb           # notebook entry point (%run ../code/fast_canonical_rebuild.py)
data/                                  # not tracked in git — see Data below
results/
  benchmark/  ghana/  lecturer/        # metrics and SHAP rankings as CSV
figures/
  benchmark/  ghana/  lecturer/        # 300 dpi PNG figures
requirements.txt
```

## Data

`data/` is excluded from version control (see `.gitignore`) because it contains
institution-level student and lecturer records that should not be made public. To
reproduce the pipeline, place the following files in `data/`:

- a benchmark dropout dataset (semicolon-delimited, with a `Target` column) — see the
  [UCI "Predict Students' Dropout and Academic Success" dataset](https://archive.ics.uci.edu/dataset/697)
- per-course grade CSVs named `<CourseCode>+...gpa...csv`
- `Teacher_Trust_Human_AI_dataset.csv` (lecturer survey responses)

## Running

```bash
pip install -r requirements.txt
python code/fast_canonical_rebuild.py
```

This regenerates every CSV in `results/` and every PNG in `figures/`. Use
`code/full_rebuild_with_optional_10fold_cv.py` instead if you also want the 10-fold
cross-validation summary for the benchmark model.

## Methodology notes

- Random seed 42 throughout.
- Stratified 80/20 train/test split; SMOTE oversampling applied to the training fold only.
- SHAP explanations (`TreeExplainer`) are computed on the held-out test fold of the
  XGBoost model.
