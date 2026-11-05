# Supervisor release verification — 5 November 2026

## Scope and results

This release has been **partially verified**. `VERIFY_RELEASE.py` completed 14/14 stored-artifact integrity checks. Scripts 7, 8, and 10 were executed successfully in this environment. These checks do **not** prove that all training, cross-validation, ablations, and survey provenance were independently reproduced.

| Component | Verification status | Evidence / qualification |
|---|---|---|
| Canonical benchmark | Stored-artifact consistency only | Script 1 has not been freshly rerun in this release. |
| Script 3 tuning | Ten-fold CSV inspected; **not freshly recomputed** | LR 0.880801, DT 0.795420, RF 0.874391, XGB 0.881451 AUC-PR. Two candidates per model; model-specific grids are not substantively equivalent. |
| Script 6 Phase 3 | Stored metrics/confusion consistent; **fresh full run timed out** | Held-out accuracy 0.884746, F1 0.813187, AUC-PR 0.890752; do not confuse with September Round-2 values. Full ten-fold rerun not independently completed here. |
| Script 7 SHAP | **Executed successfully** | Per-student signed one-hot contributions summed before absolute-value averaging. Top features: second-semester approved 1.347869, fees 0.530085, first-semester approved 0.438085, Course 0.279954. |
| Script 8 inference | **Executed successfully** | Ten-fold corrected variance factor 1/10 + 1/9 = 0.211111; 36-test Holm family. Corrected p-values depend on the specific current fold outputs, not historical reported p-values. |
| Script 10 lecturer | **Executed successfully** | N=350, clarity–trust r=0.612830, nine-predictor OLS R²=0.580103. This nine-predictor sensitivity model is **not** the same as the thesis eight-predictor specification. |
| Paired bootstrap | Existing outputs only | 600 resamples, below supervisor's requested >=2,000; not adequate to claim requested audit completed. |
| Lecturer provenance | **Unresolved** | Original form export, invitation denominator, sampling frame, ethics approval, dashboard/explanation exposure dates and repeated open-response provenance must be supplied by the researcher. Numerical reproducibility is not evidence of authentic data collection. |

## Thesis changes required before clearance

1. Cite the actual release commit only after confirming its availability in Git. Do not cite the older nonexistent `2a60aaf...` hash.
2. Do not mix the September verifier's Phase 3 result (accuracy ~0.881) with the November rerun CSV (~0.885); explain their separate source snapshots and rerun conditions.
3. State that the 10-fold tuning audit uses **two model-specific candidates** per family; it is an equal-count audit, not proof of equivalent optimization budgets or model superiority.
4. Correct SHAP source-feature ranking and replace encoded-level plots if the figures are still encoded-level.
5. Clarify whether the main lecturer regression has eight predictors and the separate sensitivity model nine; report the correct degrees of freedom for each.
6. Keep all survey conclusions provisional until the original collection records and ethics documentation are checked.
7. Complete a clean-environment end-to-end rerun, 2,000+ bootstrap resamples, and document all data/library hashes before declaring the repository **fully verified**.

## Privacy and release restriction

The supplied ZIP includes lecturer and institutional case-level files. **Do not push or publish them publicly** until data-release authorization is documented. The archive is intended for local supervisor review.
