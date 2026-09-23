# FINAL RUN STATUS — ALBERT DWAMENA-POKU

Anchor: REM-ALBERT_DWAMENA_POKU-2026-08-26

## Execution status
- Script 0: COMPLETE — schema/data gate passed (student 4,424; lecturer 350; institutional 320).
- Script 1: COMPLETE — canonical models, split, predictions, fold scores and artifacts regenerated.
- Script 2: COMPLETE — held-out figures generated from Script 1 artifacts; no refit.
- Script 3: PASS (execution validation) — equal-tuning audit executes; 3-fold validation mode was used in this environment because the full 10-fold grid exceeded the session runtime. The production configuration remains 10 folds.
- Script 4: COMPLETE — multi-seed stability analysis completed.
- Script 5: COMPLETE — SHAP generated from Script 1 canonical XGBoost; no refit.
- Script 6: COMPLETE — engineered-XGBoost Phase 3 evaluated using explicit final-run settings: FEATURE_REFINEMENT_K=20; reg_alpha=0.1; reg_lambda=2.0; gamma=0.1; min_child_weight=3.0. Same Script 1 held-out split was reused.
- Script 7: COMPLETE — categorical SHAP audit consumed Script 1 artifacts; no refit.
- Script 8: COMPLETE — dependence-corrected tests consumed Script 1 fold artifacts; no refit.

## Phase 3 held-out result
Accuracy=0.878; Precision=0.831; Recall=0.778; F1=0.804; ROC-AUC=0.915; AUC-PR=0.888. Confusion counts: TN=556, FP=45, FN=63, TP=221.

## Phase 3 10-fold CV
Accuracy=0.867±0.013; Precision=0.820±0.030; Recall=0.755±0.032; F1=0.785±0.022; ROC-AUC=0.908±0.014; AUC-PR=0.867±0.018.

## Interpretation
Phase 3 is now an evaluated ablation, but it does not improve on the canonical XGBoost from Script 1 (held-out Accuracy=0.888, F1=0.818, ROC-AUC=0.934, AUC-PR=0.905). It should therefore be reported as a negative ablation result rather than as an improved model.

## Provenance note
The Phase 3 parameter values above are newly made explicit for this final rerun. They must not be described as having been part of an earlier approved methodology unless that approval is documented. Scripts 2, 5, 7 and 8 retain the no-refit artifact contract.
