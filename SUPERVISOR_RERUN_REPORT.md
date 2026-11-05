# Dr. Osei — completed local rerun report (5 November 2026)

## Execution and provenance
This release uses the uploaded repository's UCI student data, Script 0 schema and Script 1 locked 80:20 split (3,539 train / 885 held out), random state 42. No test-set examples are used to fit preprocessing, feature selection or SMOTE. Results are new executions in this environment; do not mix them with earlier repository versions. Python packages differ from those used to serialize some old models; the fresh experiments fit new models.

## Fully executed
- Script 3: full configured **10-fold** StratifiedKFold equal-budget GridSearchCV, two candidates per classifier; average precision scoring. This is a tuning audit, not nested cross-validation. Grids have equal *candidate count* but differ by model-specific parameter, and should not be described as an identical hyperparameter space.
- Script 4: five independent stratified 80:20 seeds (11, 23, 42, 71, 101) for Logistic Regression, Decision Tree, Random Forest and XGBoost; freshly refitted rather than loading incompatible pickles.
- Script 6: engineered Phase 3 XGBoost with 20-feature mutual-information selection, training-only SMOTE and explicit regularization; fresh held-out and 10-fold training CV outputs.
- Script 11: canonical, no-SMOTE, regularization-only, feature-selection-only, full Phase 3, Round-1 depth-4 sensitivity, approved/enrolled ratio sensitivity, and LR/RF SMOTE-only / class-weight-only / both comparisons.
- Earlier Scripts 5, 7, 8, 9 and 10 and their outputs retained from partial audit (SHAP, corrected tests, bootstrap, thresholds, subgroup and lecturer analyses).

## Interpretation and constraints
- Phase 3 held-out: accuracy 0.884746, precision 0.847328, recall 0.781690, F1 0.813187, ROC-AUC 0.916630, AUC-PR 0.890752. Canonical XGB in same-data Script 11: accuracy 0.888136, F1 0.817680, AUC-PR 0.905446. **Engineered Phase 3 did not outperform canonical XGBoost.**
- Script 11 is a single held-out-split sensitivity study. It is not an independently tuned or nested-CV comparison. Its ratio feature uses second-semester approved/enrolled values; availability at the proposed early-warning decision time must be verified before operational use.
- Feature-selection and regularization arms are a component ablation; there is no independent 'hyperparameter tuning only' arm because the original XGBoost canonical settings were already fixed and the equal-effort tuning audit is separate.
- The class-weight-plus-SMOTE combination is deliberately reported but may be redundant and should not be assumed superior.
- Subgroup results are exploratory, not proof of algorithmic fairness. Sensitive or small subgroups require additional safeguards and confidence intervals.
- No external Ghanaian deployment validation, prospective trust intervention, or approval of institutional access is established by these reruns.

## File index
- `results/script3_equal_tuning_audit.csv`: full 10-fold model tuning audit
- `results/script4_multi_seed_scores.csv`, `results/script4_multi_seed_summary.csv`: five-seed comparisons
- `results/script6_phase3_heldout_metrics.csv`, `results/script6_phase3_fold_scores.csv`, `results/script6_phase3_status.csv`: fresh Phase 3
- `results/script11_supervisor_experiments.csv`: component ablations, LR/RF balancing, depth and ratio sensitivity
- `results/script9_*.csv`: previously completed bootstrap, threshold and subgroup diagnostics
- `results/script10_*.csv`: lecturer analysis

The thesis's Table 4.6 should only be replaced after checking that the table uses this same repository version, label definition and split. A clean run on the target pinned environment is recommended before formal submission.
