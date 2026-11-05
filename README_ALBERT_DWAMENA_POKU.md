# Albert Dwamena-Poku — XGBoost/SHAP reproducibility repository

**Study:** *An XGBoost-SHAP Explainability Framework for Teacher-Centred Classroom Decision Support in Ghanaian Higher Education*  
**Student:** Albert Dwamena-Poku (PG3586125) · **Supervisor:** Dr Eric Opoku Osei

## What this repository actually supports

This repository contains a **public UCI higher-education benchmark** comparison of Logistic Regression, Decision Tree, Random Forest and XGBoost, SHAP explanations of the canonical XGBoost model, Phase 3 sensitivity/ablation experiments, and a **separate** Ghanaian institutional data-readiness audit. The Ghanaian grade extract **does not support a validated early-warning prediction model** because coursework contributes to the final-score target. The lecturer survey is a cross-sectional analytical file, **not proof of a SHAP exposure experiment or a causal trust effect**.

**Important:** This package consolidates the completed 5 November 2026 rerun release with the latest uploaded source snapshot. It is **not a fresh end-to-end verification of every experiment against the newly uploaded archive**. See `VERIFICATION_AND_PROVENANCE.md` for verified checks, version differences and unresolved provenance questions. No claim of supervisor sign-off is made.

## Run locally

From the repository root, create a Python environment, install compatible dependencies from `requirements.txt` and execute the scripts in order. `ALBERT_DWAMENA_POKU_SCRIPT_0_SCHEMA_GATE.py` reads files under `data/` via relative paths. The full training/audit workflow is computationally expensive; allow substantial time and do not set smoke-test environment variables when reporting a production 10-fold audit.

```bash
python ALBERT_DWAMENA_POKU_SCRIPT_0_SCHEMA_GATE.py
python ALBERT_DWAMENA_POKU_SCRIPT_1_CANONICAL_FIT.py
python ALBERT_DWAMENA_POKU_SCRIPT_2_HELDOUT_FIGURES.py
python ALBERT_DWAMENA_POKU_SCRIPT_3_EQUAL_TUNING_AUDIT.py
python ALBERT_DWAMENA_POKU_SCRIPT_4_MULTI_SEED_STABILITY.py
python ALBERT_DWAMENA_POKU_SCRIPT_5_SHAP_FROM_CANONICAL.py
python ALBERT_DWAMENA_POKU_SCRIPT_6_PHASE3_GATED_ABLATION.py
python ALBERT_DWAMENA_POKU_SCRIPT_7_CATEGORICAL_SHAP_AUDIT.py
python ALBERT_DWAMENA_POKU_SCRIPT_8_DEPENDENCE_CORRECTED_TESTS.py
python ALBERT_DWAMENA_POKU_SCRIPT_9_HELDOUT_DIAGNOSTICS.py
python ALBERT_DWAMENA_POKU_SCRIPT_10_LECTURER_STATISTICS.py
python ALBERT_DWAMENA_POKU_SCRIPT_11_SUPERVISOR_EXPERIMENTS.py
```

Scripts 9–11 are additional supervisor-requested diagnostics. Existing committed CSVs in `results/` and figures must be linked to the precise code/data snapshot used to produce them. Never combine metrics from different model configurations in one table without labeling them.

## Reproduced results supplied with this release

- Canonical XGBoost: accuracy **0.888**, F1 **0.818**, AUC-PR **0.905**; canonical confusion matrix **TN 564 / FP 37 / FN 62 / TP 222**.
- Fresh Phase 3 rerun in the 5 November completed release: accuracy **0.884746**, precision **0.847328**, recall **0.781690**, F1 **0.813187**, ROC-AUC **0.916630**, AUC-PR **0.890752**; confusion matrix **TN 561 / FP 40 / FN 62 / TP 222**. This is a negative ablation relative to canonical XGBoost.
- Full 10-fold equal-candidate audit (`results/script3_equal_tuning_audit.csv`): LR **0.880801**, DT **0.795420**, RF **0.874391**, XGBoost **0.881451** in best CV AUC-PR. The audit tests **two candidate settings per model**, which is not equivalent to exhaustively equalising model complexity or search space.
- Correct source-level SHAP uses `mean(abs(sum(encoded-level SHAP values for a student)))`; see `results/script7_shap_ranking_source_features.csv`. In this release curricular units approved in semester 2 rank first; tuition fees rank second; Course ranks fourth.
- The dependence-corrected CV analysis uses **1/9** as the test/train ratio for 10 folds and Holm correction across 36 tests. See `results/script8_dependence_corrected_tests.csv`.
- Paired bootstrap results in `results/script9_paired_bootstrap.csv` use **600 resamples**, not 2,000; they are preliminary and do not satisfy the Round 2 report's ≥2,000 requirement.

## Data governance and limitations

The uploaded archives contain individual-level lecturer and institutional files. **Do not push these files to a public remote without documented authorization, ethical clearance and disclosure review.** Local packaging here does not establish public-release permission. The current lecturer CSV has 350 records and a different SHA-256 checksum from the Round 2 report's cited earlier file; see the provenance report. The original raw form export, invitations/response rate, dashboard/exposure procedure, and an explanation of repeated open-ended answers remain necessary before interpreting lecturer findings as independently authenticated survey evidence.

## Files and folders

- `data/`: supplied datasets; review access permissions before publishing.
- `artifacts/`: split indices, model artifacts and predictions; restricted case-level diagnostic files must not be posted publicly.
- `results/`: metrics, tuning, SHAP, bootstrap, subgroup and lecturer analysis outputs.
- `figures/`: analysis figures; historical Round-1 figures should not be cited as current.
- `VERIFICATION_AND_PROVENANCE.md`: verification checklist, unresolved discrepancies and source-snapshot lineage.
- `requirements.txt`: dependency specification; record an exact `pip freeze` and Python version for final reproducibility.

## Citation

For a thesis or manuscript, cite **the actual local Git commit** included in this ZIP and its tag. This local commit is **not automatically published to GitHub**. Do not cite the historical nonexistent `2a60aaf...` commit. Run `git rev-parse HEAD` in the extracted repository to obtain the exact commit hash.
