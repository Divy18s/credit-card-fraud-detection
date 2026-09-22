# Credit Card Fraud Detection — ML · DL · GNN

> **Complete.** 72 time-split runs · 12 split-ablation runs · 6 bootstrap configs · 6 pre-registered replication runs.
> **Read:** [`reports/report.pdf`](reports/report.pdf) (full study) · [`reports/why_chronological_split.pdf`](reports/why_chronological_split.pdf) (protocol rationale)

Classical ML, deep learning, GAN oversampling, ensembles and GNNs compared on the
[Kaggle ULB dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 frauds, 0.173%) under one chronological protocol.

**Champion:** Random Forest + SMOTE-ENN — PR-AUC **0.795**, F1 **0.848**, 97.5% precision at 75% recall, **1 false positive** in 42,722.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Kaggle token → `~/.kaggle/access_token` or `~/.kaggle/kaggle.json`.
GPU: `.venv` ships `torch 2.13.0+cu130` (RTX 5050, sm_120). XGBoost/CatBoost use CUDA; LightGBM is CPU.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Download | `python -m src.data.make_dataset` | `data/raw/creditcard.csv` |
| 2. EDA | `notebooks/01_eda.ipynb` | `reports/figures/` |
| 3. Train | `python -m src.models.train_classical` / `train_deep` / `train_gnn` / `train_ensembles` | `reports/results_classical.csv` |
| 4. Tune | `python -m src.models.tune_optuna` | `reports/optuna_best_params.json` |
| 5. Split ablation | `python -m src.models.ablation_random_split` | `reports/results_random_split.csv` |
| 6. SHAP | `python -m src.models.explain_shap` | `reports/shap_summary.txt` |
| 7. Bootstrap CIs | `python -m src.models.bootstrap_ci` | `reports/bootstrap_ci.csv` |
| 8. Replication | `python -m src.models.train_classical --split sang` | `reports/results_sang_split.csv` |
| 9. Demo | `.venv\Scripts\streamlit run app/streamlit_app.py` | browser UI |
| 10. Report | `notebooks/02_final_report.ipynb` | full submission report |

## Results — chronological 70/15/15, 72 runs

Test = latest 15% (42,722 txns, 52 frauds), untouched distribution. Threshold tuned on validation.
Full ledger: `reports/results_classical.csv`.

| Rank | Model | Strategy | PR-AUC | F1 | MCC | FP | FN |
|---|---|---|---|---|---|---|---|
| 1 | Random Forest | SMOTE-ENN | **0.795** | 0.848 | 0.855 | 1 | 13 |
| 2 | Random Forest | SMOTE | 0.790 | 0.848 | 0.855 | 1 | 13 |
| 3 | **RF+MLP+LSTM** | **cross-family stack** | **0.785** | 0.835 | 0.844 | 1 | 14 |
| 4 | CatBoost | GAN | 0.783 | 0.848 | 0.855 | 1 | 13 |
| 5 | BiLSTM | class_weight | 0.781 | 0.839 | 0.844 | 2 | 13 |
| 6 | LSTM | class_weight | 0.780 | 0.813 | 0.821 | 2 | 15 |
| 7 | XGBoost | SMOTE-ENN | 0.777 | 0.813 | 0.815 | 5 | 13 |
| 8 | LightGBM | GAN | 0.775 | 0.822 | 0.832 | 1 | 15 |
| — | LSTM ensemble (5 seeds) | mean | 0.775 | 0.831 | 0.843 | 0 | 15 |
| — | **GraphSAGE** (k-NN graph) | class_weight | 0.756 | 0.796 | 0.801 | – | – |
| — | MLP | raw / smote / gan | 0.719–0.775 | — | — | — | — |
| — | GATv2 | class_weight | 0.745 | – | – | – | – |
| — | FT-Transformer | class_weight | 0.681 | 0.760 | 0.760 | 10 | 14 |
| — | AutoEncoder | anomaly (legit-only) | 0.584 | 0.660 | 0.660 | 19 | 17 |

*Sequence models drop the first 7 test rows lacking an 8-step window (fraud counts unaffected, 52/52).*

## Uncertainty — the headline result

Stratified bootstrap ×1,000 per tier gives CI widths **0.20–0.24**. **All 15 pairwise comparisons overlap.**
At 52 test frauds **one transaction is worth ±0.020 PR-AUC**, so no model is separable from another —
the honest output is a shared **~0.72–0.89 band, not a ranking**.

Paired bootstrap, RF vs XGBoost: **+0.0187, 95% CI [−0.0119, +0.0515]** — includes zero.

## Why a chronological split

> **The core claim.** This is a **validity** argument, not a performance one. A random split trains
> the model on transactions that occurred *after* the ones it is scored on. No deployed system can
> do that, so the number it produces answers a question that has no operational counterpart.
> **Chronological evaluation would be correct even if both protocols scored identically.**

Full write-up with sources: [`reports/why_chronological_split.pdf`](reports/why_chronological_split.pdf).

### What a random split actually does

Shuffling assigns rows to partitions by coin flip. Transaction #250,000 (hour 45) can land in
training while transaction #40,000 (hour 7) lands in test. The model is then fitted on a fraud
pattern that emerged *late* and graded on transactions from *earlier* — the exact inverse of
deployment, where you have only the past and must score the future.

Three distinct channels follow from that. This project measures the first two and deliberately
excludes the third:

| # | Channel | In this project |
|---|---|---|
| 1 | **Temporal ordering destroyed** — future patterns inform training | measured, ±0.018 |
| 2 | **Duplicate transactions separated** across train/test | measured, 517 rows |
| 3 | **Resampling applied before the split** (common in published work) | excluded by construction — our resampling is fitted inside the pipeline |

Because channel 3 is excluded, **every effect reported here is a floor, not an estimate** of what
the protocol is worth in the wider literature.

### Channel 1 — the drift it hides

ULB spans 48 hours. Even inside that window the distribution moves, so a model fitted on the
early period is genuinely solving a different problem from one fitted on a shuffled sample:

| Partition | Window (h) | n | Frauds | Fraud rate | Mean amount |
|---|---|---|---|---|---|
| Train | 0.00 – 36.92 | 199,364 | 384 | **0.193%** | 89.77 |
| Validation | 36.92 – 42.04 | 42,721 | 56 | **0.131%** | 97.52 |
| Test | 42.04 – 48.00 | 42,722 | 52 | **0.122%** | 72.54 |

The fraud prior falls **37% relative** train → test and mean transaction amount **19%**. The window
also covers two full day/night cycles with a **35.5×** ratio between peak and trough hourly fraud
rates (02:00 peaks at 1.71%).

Shuffling erases all of it — it makes the train and test distributions identical *by construction*,
which is precisely the condition production never provides. This matters doubly for PR-AUC, whose
baseline is the positive rate: change the prevalence of the test window and you change the floor
the metric is measured against.

### Channel 2 — shuffling separates duplicate transactions

ULB contains **1,081 exact duplicate rows** in 773 groups. Every group shares a single `Time`
value, so sorting chronologically places each copy next to its twin and they always land together:

| | Train→Test | Train→Val | Val→Test | Total leaked |
|---|---|---|---|---|
| **Chronological** | 0 | 0 | 0 | **0** |
| **Random** | 249 | 207 | 61 | **517** |

Under shuffling the model is fitted on a row and then scored on a byte-identical copy of it. That
is memorisation reported as generalisation. The chronological split is immune **for free** — no
rows deleted, and the card-testing signal those duplicates carry (14.3% fraud rate among €0.00
duplicates, 4.6% among €0.01–2.00, against a 0.173% base rate) stays in the data.

### What we measured

Same data, same features, same code, same hyperparameters, same threshold rule — only the split
assignment changes:

| Config | chronological | random | random − chrono |
|---|---|---|---|
| XGBoost + class_weight | 0.7688 | 0.8372 | +0.0684 |
| CatBoost + GAN | 0.7804 | 0.8399 | +0.0595 |
| MLP raw | 0.7794 | 0.8267 | +0.0473 |
| XGBoost + SMOTE-ENN | 0.7766 | 0.8104 | +0.0338 |
| RF + SMOTE-ENN | 0.7947 | 0.8120 | +0.0173 |
| LSTM + class_weight | 0.7803 | 0.7965 | +0.0162 |
| GraphSAGE + class_weight | 0.7389 | 0.6948 | −0.0441 |

*70/15/15. Mean +0.0283, positive in 6 of 7.*

**Pooled over all 10 paired runs (both cut points): +0.0182, 95% CI [−0.0084, +0.0448], 7/10 positive.**

Stated honestly: the interval **includes zero** (*t* = 1.55, sign test *p* = 0.172) and the estimate
is unstable across cut points (+0.028 at 70/15/15, −0.005 at 66/14/20). **These data establish the
direction, not a reliable magnitude** — and the validity argument above does not depend on the
magnitude.

### What the random number cannot be used for

A random-split score answers *"how well would this model rank transactions if it had already seen
the future?"* That estimate cannot size a review queue, set an alert threshold, or forecast analyst
caseload, because the condition it assumes never holds. The chronological score can — it is
measured under exactly the information a deployed model would have.

### Three objections, answered

- *"Random gives more test frauds (74 vs 52), so tighter estimates."* True, and precision does
  improve. But precision on an invalid estimand buys nothing — and if you want more test frauds you
  can widen the tail chronologically instead (66/14/20 yields 75).
- *"Everyone uses random splits, so you can't compare."* That is the point: the comparison was never
  valid. [arXiv:2506.02703](https://arxiv.org/abs/2506.02703) names inadequate temporal validation as
  one of four recurring flaws in exactly this literature.
- *"48 hours is too short for real drift."* The prior still falls 37% relative inside it. And a short
  horizon makes our figure **conservative** — a deployment gap of weeks would be larger, not smaller.

### What the literature says

Cited for the **principle**, not for number-matching (architectures, search budgets and
preprocessing all differ — see [Benchmark replication](#benchmark-replication--the-published-split)):

- **[Dal Pozzolo et al. 2018](https://re.public.polimi.it/bitstream/11311/1044896/1/08038008.pdf)** (IEEE TNNLS 29(8)) — **the ULB dataset's own authors.** They formalise fraud detection around *concept drift*, class imbalance and *verification latency*. Two of those three are inherently temporal; a shuffled split represents neither.
- **[Sang 2026](https://publications.eai.eu/index.php/ismla/article/view/12078)** — the only study we found running ULB under *both* protocols. Concludes random stratification *"may provide optimistic estimates … when earlier and later transactions are mixed across training, validation, and test partitions"*, and measures the gap at **+0.091** on its own model.
- **[arXiv:2506.02703](https://arxiv.org/abs/2506.02703)** — audit of published ULB pipelines. Lists *"inadequate temporal validation for transaction data"* among four recurring flaws, and demonstrates the stakes with a deliberately leaky minimal MLP that reaches **99.9% recall** and beats far more sophisticated published methods.
- **[arXiv:2603.06632](https://arxiv.org/abs/2603.06632)** — strict chronological split because it *"reflects a realistic deployment setting in which models are trained on historical data and evaluated on future transactions."* Different dataset (Elliptic); cited for principle only.

## Benchmark replication — the published split

[Sang 2026](https://publications.eai.eu/index.php/ismla/article/view/12078) reports ULB chronological **0.7902** / random **0.8815**.
Its 66/14/20 partition reproduces exactly from the public CSV (187,972/39,873/56,962; 368/49/75 frauds).
Question and configs fixed in [`reports/PREREGISTRATION_sang_split.md`](reports/PREREGISTRATION_sang_split.md), committed **before** the run.

| Config | our 70/15/15 | their 66/14/20 | Δ split | vs 0.7902 |
|---|---|---|---|---|
| XGBoost + class_weight | 0.7688 | **0.8055** | +0.0367 | +0.0153 (**0.37 SE**) |
| XGBoost + SMOTE-ENN | 0.7766 | **0.8102** | +0.0336 | +0.0200 (0.48 SE) |
| RF + SMOTE-ENN | 0.7947 | **0.8224** | +0.0276 | +0.0322 (0.77 SE) |

- **Replication, not improvement** — 0.37 SE from the published value, same 57/75 frauds recovered (FN=18), 5 false positives vs 9. Primary protocol stays 70/15/15.
- **Cut point is first-order** — fractions alone moved PR-AUC **+0.033**, larger than the pooled protocol effect. Our 0.795 assumes a 15% tail; 20% gives 0.822.

## Explainability (SHAP)

TreeExplainer on the champion over 3,052 test transactions. **Top drivers: V12, V14, V4, V10, V11, V17** —
directions match the EDA correlation signs exactly. Champion bundle (`{pipeline, threshold, feature_order}`):
`src/models/artifacts/champion_rf_smote_enn.joblib`.

## Demo app

```powershell
.venv\Scripts\streamlit run app/streamlit_app.py
```

Single transaction (live probability, verdict, SHAP waterfall) · batch CSV scoring · leaderboard with CIs.
Samples from the held-out test slice only.

## Key findings

- **Hybrid sampling wins for trees** — SMOTE-ENN takes the champion slot (Ahmed et al. 2025)
- **GAN-synthesised fraud is top-tier for boosting** — CatBoost+GAN matches RF's error profile; repairs LightGBM's raw weakness
- **Cross-family stacking works, same-family doesn't** — RF+MLP+LSTM (0.785) beats every deep model; a 5-seed LSTM ensemble (0.775) doesn't beat its own members
- **GNN on a k-NN graph ≈ plain MLP** (SAGE 0.756). The AUC-PR ≈ 0.89 figure is [HG-AE](https://arxiv.org/abs/2410.08121)'s and needs real cardholder/merchant graphs — a *data* gap, not architecture. GATv2 (0.745) adds cost, not accuracy
- **FT-Transformer (0.681) loses to boosting and to the MLP** — matches the tabular-data literature
- **Tuning repairs, it doesn't lift** — LightGBM 0.537→0.753, XGBoost 0.714→0.769, RF 0.749→0.782; every tier champion stays within ±0.01 of its untuned best
- **Depth doesn't help** — DNN (512-256-128-64) ties the small MLP (0.7752 vs 0.7746); Decision Tree 0.405 vs its own RF 0.795
- **More steps hurt GNNs here** — 300-step SAGE without early stopping *degrades* to 0.707; validation AP rises while test AP falls. Temporal drift, not over-smoothing (depth fixed at 2)

## Protocol anomalies (kept as findings)

- **`scale_pos_weight` = imbalance ratio (519) collapses boosted trees** — LightGBM drops to **0.009**. Validation selection from `{1, 4, 16, √ratio, 64, ratio}` rejects weighting entirely (spw=1) and recovers 0.537. Both variants reported
- **CatBoost + random undersampling degrades** (0.670) — ordered boosting needs volume; 768 rows is too few
- **GaussianNB collapses** (~0.08) — independence assumption plus extreme prior
- *Erratum:* an early run cached the GAN generator across both split protocols, inflating random-side CatBoost+GAN to 0.877. Cache now keyed by a hash of the training frauds; ablation fully re-measured

## References

- **Sang, V. N. T.**, *Enhancing Credit Card Fraud Detection under Severe Class Imbalance*, EAI Endorsed Trans. ISMLA 3, 2026 — the protocol-matched benchmark
- **Dal Pozzolo et al.**, *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*, IEEE TNNLS 29(8), 2018 — the ULB dataset's own authors
- *Data Leakage and Deceptive Performance*, arXiv:2506.02703, 2025 — four recurring flaws in ULB pipelines
- Ahmed et al., *Ensemble ML with hybrid data sampling*, MLWA, 2025
- Anand & Chakrabarty, *Credit Card Fraud Detection and Credit Risk Analysis*, SSRN, 2025
- Singh et al., *Heterogeneous Graph Auto-Encoder*, arXiv:2410.08121
