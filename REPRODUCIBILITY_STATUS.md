# Reproducibility audit — Dr. Osei feedback

## Verified in this working environment
- Script 0 repaired and executed; datasets validated by SHA-256 checksums.
- Script 5 executed; encoded per-student SHAP matrix and additivity saved.
- Script 7 executed; signed one-hot SHAP values aggregated within each student before taking mean absolute attribution; source-variable figures regenerated.
- Script 8 executed; Nadeau–Bengio ratio set to 1/(k-1), equal to 1/9 at ten folds; Holm family explicitly all 36 pairwise metric tests.
- Script 9 executed on **uploaded repository predictions**: threshold sweep, paired bootstrap (600 replicates), subgroup diagnostics, false-negative audit.
- Script 10 executed on **uploaded lecturer questionnaire**: alpha, clarity-trust correlation, exploratory OLS.

## Code updated but not fully re-executed
- Script 3: same two-point search budget for each model, original LR solver/iterations, configured 10-fold audit. The pre-existing CSV must NOT be represented as the newly verified 10-fold output until rerun.
- Script 4: five-seed stability now includes all four classifiers. Run before citing its new output.
- Script 6: original Phase-3 pipeline preserved; **not rerun in this revision**. Existing metrics remain historical and must not be described as freshly reproduced.

## Remaining supervisor requests (not implemented or not validated)
- Full Phase-3 five-component ablation, LR/RF SMOTE-versus-class-weight factorial, depth-4 Round-1 sensitivity, approved/enrolled ratio sensitivity.
- Admission-grade SHAP dependence, actionable-only waterfall, calibration and fairness validation.
- Full ten-fold audit and all-model multi-seed and Phase-3 reruns in a version-matched environment.
- Original fitted models warn they were serialized under scikit-learn 1.9.1 but this environment uses 1.8.0. Refit and save under a pinned validated environment before asserting reproducibility.

## Data and provenance warnings
- Uploaded held-out XGBoost has **63 false negatives**, not 62. The supplied archive is NOT identical to the previously discussed locked repository. Do not mix metrics, tables or figures across versions.
- Raw data are bundled locally for reruns but ignored by Git; DATA_SHA256SUMS.txt is committed. Lecturer records may include sensitive responses; obtain permissions before public release.
- Git tag documents this patch revision, **not** proof that all supervisor analyses passed.
