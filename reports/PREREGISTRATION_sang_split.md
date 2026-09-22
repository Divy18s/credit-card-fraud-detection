# Pre-registration: replication check against Sang (2026) chronological split

**Written and committed BEFORE the experiment was run.** Git history is the evidence.

## Question (fixed in advance)

When the chronological split is set to Sang's 66/14/20 rather than this project's
70/15/15, does the champion still match the published PR-AUC of 0.7902?

## Why

The head-to-head in the report compares this project's RF + SMOTE-ENN (70/15/15,
52 test frauds) against Sang's Optuna-tuned cost-sensitive XGBoost (66/14/20,
75 test frauds). Two things are confounded: the split fractions differ, and the
model families differ. This check removes the first confound and, by running our
own XGBoost, addresses the second.

Sang's split reproduces exactly from the public CSV: 66/14/20 on time-sorted,
non-deduplicated rows gives 187,972 / 39,873 / 56,962 with 368 / 49 / 75 frauds,
matching the paper to the unit.

## Configurations to run (fixed in advance, no others)

1. `xgboost + class_weight`   - closest analogue to Sang's cost-sensitive XGBoost
                                (validation-selected weight, no resampling)
2. `xgboost + smote_enn`      - this project's best XGBoost
3. `random_forest + smote_enn`- this project's champion, for continuity

## Interpretation agreed in advance

- XGBoost near 0.79  -> independent replication of Sang's result.
- XGBoost near 0.77  -> the difference is their cost-sensitive tuning, not the
                        protocol. Reported as such.
- Champion moves substantially -> the champion is split-fraction sensitive;
                        that goes in Limitations.

## Guardrails

- This is a ROBUSTNESS CHECK reported ALONGSIDE the primary 70/15/15 result,
  never replacing it. The primary result does not change.
- The outcome is reported whichever way it falls. Selecting whichever split
  flatters the project would be exactly the best-of-k inflation this report
  quantifies at +0.059, and is forbidden here.
- 75 test frauds vs 52 narrows the CI by only ~17% (SE scales as 1/sqrt(n)).
  This cannot resolve the statistical tie and is not expected to.
