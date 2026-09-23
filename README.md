# Updated XGBoost–SHAP Reproducibility Package

All requested figures are regenerated from a single canonical run. Figures are saved as 300-dpi PNG files.

## Input data (in `data/`)
- `ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv` - benchmark UCI dropout dataset
- `ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv` - Ghanaian institutional dataset
- `ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv` - lecturer survey

## Manuscript figures
- Figure 4.2: `figures/benchmark/Figure_4_2_model_performance_comparison.png`
- Figure 4.3: `figures/benchmark/Figure_4_3_ROC_curves.png`
- Precision–Recall curves, individual and combined confusion matrices, and SHAP global/beeswarm/waterfall figures are in `figures/benchmark/`.
- Ghanaian institutional equivalents are in `figures/ghana/`.

## Canonical settings
Random seed 42; stratified 80:20 split; SMOTE applied only to training data; fixed tuned model parameters; benchmark 10-fold CV summary included.

## Ghanaian early-warning model
Intervention is defined as Overall_Score < 50. Exam_60 and Overall_Score are excluded from inputs to prevent target leakage.

## Lecturer data
The supplied 350-row file is analysed separately. Check `results/lecturer/record_status_counts.csv` before making collection-status claims.
