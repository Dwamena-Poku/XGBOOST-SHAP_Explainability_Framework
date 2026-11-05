# Repository verification and provenance — 5 November 2026

## Source snapshots

1. User's newly uploaded `Albert_XGBoost_SHAP(1).zip`: HEAD `62ef0fd815f322708bc32f55c79b88540a29892f`; working tree includes deletions, renames and modified data. Its `results/script6_phase3_heldout_metrics.csv` records accuracy **0.881356**, F1 **0.807339**, AUC-PR **0.886606**; its Script 3 audit reports XGBoost CV AUC-PR **0.866391** and RF **0.876134**. These are **not identical** to the earlier completed rerun release.
2. Previous completed supervisor rerun release: HEAD `a1d836c867dd30b7100be40ec802331ad917b67e`; contains 10-fold audit, corrected SHAP aggregation, diagnostics and Phase 3 rerun. This is the basis for this updated release, not a silent replacement of the uploaded snapshot's results.
3. Round 2 verification document dated 24 September 2026 examines still another earlier HEAD (`b639e69`) and reports different Phase 3 results. Its numbers must not be mixed with this release.

## Checks actually performed in this revision

- Inspected the uploaded ZIP's Git HEAD, dirty working-tree status, Script 0 and key CSVs.
- Compared Script 3, Script 6 and source-level SHAP results with the earlier completed rerun package.
- **Executed Script 0** successfully against the data in the consolidated package: UCI **4,424** rows, lecturer **350**, institutional **320**.
- Verified the latest lecturer file has 350 rows, a timestamp range of **15 June–14 July 2026**, and only **six distinct responses** in the inspected 'How did SHAP explanations influence your trust' open-ended column. This is a provenance concern, not evidence of misconduct.
- Confirmed the completed rerun package contains a 10-fold audit and a 36-test Holm family in its result files. These CSVs were not independently regenerated during this update.
- The attempted increase of Script 9 bootstrap resamples from 600 to 2,000 exceeded the execution time budget; the script and output are retained at **600**, explicitly labeled as outstanding.

## Checks that remain open before supervisor sign-off

- Re-run the complete training, 10-fold tuning and Phase 3 chain on **the final selected data snapshot** in a clean environment; record exact Python/library versions, timestamps and hashes. Confirm whether the uploaded source's Phase 3 metrics or the prior completed rerun metrics are authoritative.
- Explain why the uploaded source's XGBoost tuning score differs from the completed release; inspect model settings, data hashes, library versions and seeds. Two-candidate tuning is limited evidence of baseline symmetry.
- Review SHAP attribution source-feature grouping and confirm regenerated bar/beeswarm figures correspond to the same model and prediction matrix.
- Review bootstrap CIs with at least **2,000** paired resamples and decide threshold/cost trade-offs before claiming operational performance.
- Obtain raw survey export with native metadata, ethics reference, sampling frame, number invited, response rate, and an auditable explanation of repeated open-ended answers. Establish what explanation/dashboard respondents saw **before** June 2026. Do not claim a measured causal effect of SHAP exposure.
- Review whether personal lecturer/institutional data may legally/ethically be distributed; keep restricted case-level outputs private.
- Reconcile thesis tables and figures to the chosen final release; explicitly disclose that the public benchmark is not Ghanaian and semester-two variables do not support within-class early warning.

## Integrity note

The archive retains the uploaded source snapshot separately **outside** the Git working tree in `SOURCE_UPLOAD_SNAPSHOT.zip` only if included by the packager. The consolidated repository does not represent an automatic push or merge to the user's remote GitHub repository.
