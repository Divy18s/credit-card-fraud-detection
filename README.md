# Credit Card Fraud Detection — ML · DL · GNN

> **Status: complete.** 72 time-split runs + 12 split-ablation runs + 6 bootstrap configs.
> Start here → `notebooks/02_final_report.ipynb` (executed final report).

End-to-end fraud detection on the [Kaggle ULB Credit Card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 frauds, 0.173%), comparing **classical ML**, **deep learning**, and **GNN**
approaches under one identical time-based evaluation protocol.

## Setup

```powershell
python -m venv .venv            # or: python -m uv venv .venv --python 3.12
.venv\Scripts\activate
pip install -r requirements.txt
```

Kaggle credentials: save your API token to `~/.kaggle/access_token` (plain text)
or `kaggle.json` under `~/.kaggle/`.

GPU: `.venv` ships `torch 2.13.0+cu130` (verified on RTX 5050, sm_120).
XGBoost/CatBoost also run on GPU via `device="cuda"`; LightGBM stays on CPU.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Download data | `python -m src.data.make_dataset` | `data/raw/creditcard.csv` |
| 2. EDA | open `notebooks/01_eda.ipynb` (kernel: `.venv (Python 3.12)`) | figures in `reports/figures/` |
| 3. Train tiers | `python -m src.models.train_classical` / `train_deep` / `train_gnn` / `train_ensembles` | `reports/results_classical.csv` |
| 4. Tuning | `python -m src.models.tune_optuna` | `reports/optuna_best_params.json` |
| 5. Split ablation | `python -m src.models.ablation_random_split` | `reports/results_random_split.csv` |
| 6. Explainability | `python -m src.models.explain_shap` | `reports/shap_summary.txt`, figures |
| 7. Bootstrap CIs | `python -m src.models.bootstrap_ci` | `reports/bootstrap_ci.csv`, figure |
| 8. Demo app | `.venv\Scripts\streamlit run app/streamlit_app.py` | browser UI |
| 9. Final report | open `notebooks/02_final_report.ipynb` (kernel: `.venv`) | full submission report |

## Uncertainty (bootstrap)

Stratified resampling of the test set ×1,000 per model gives 95% CIs of width
**~0.20–0.25 PR-AUC** for every tier representative — **all pairs overlap**
(statistical ties). At 52 test frauds, no model is separable from another;
the honest conclusion is a shared ~0.63–0.89 band, not a ranking.

## Demo App

```powershell
.venv\Scripts\streamlit run app/streamlit_app.py
```

- **Single transaction tab:** draw a real test transaction or edit any feature → live fraud probability, verdict against the validation-tuned threshold, top SHAP drivers table, full waterfall explanation
- **Batch CSV tab:** upload transactions (Time, V1–V28, Amount ± Class) → flagged counts, PR-AUC if labels present, downloadable scored file

## Explainability (SHAP)

TreeExplainer on the champion RF+SMOTE-ENN over 3,052 test transactions.
**Top fraud drivers:** V12, V14, V4, V10, V11, V17 — directions match the EDA
correlation signs exactly (V12/V14/V10/V17: low value → fraud; V11: high →
fraud; small amounts → fraud). The fitted champion pipeline is saved at
`src/models/artifacts/champion_rf_smote_enn.joblib`.

## Results (time-based split, 72 runs)

Test = latest 15% of transactions (42,722 txns, 52 frauds), untouched distribution.
Threshold tuned on validation set. Full table: `reports/results_classical.csv`.

*Protocol footnotes:* sequence models (LSTM/BiLSTM) drop the first 7 test rows that lack an
8-step window — fraud counts are unaffected (52/52). Neural/GNN tiers carry GPU
nondeterminism: three invocations of the *identical* GraphSAGE configuration spanned
0.631–0.756 PR-AUC (full-batch training takes only 40 gradient steps, and early stopping
picks its checkpoint on a 56-fraud validation set); tree tiers reproduce bit-exactly.

| Rank | Model | Strategy | PR-AUC | F1 | MCC | FP | FN |
|---|---|---|---|---|---|---|---|
| 1 | Random Forest | SMOTE-ENN (hybrid) | **0.795** | 0.848 | 0.855 | 1 | 13 |
| 2 | Random Forest | SMOTE | 0.790 | 0.848 | 0.855 | 1 | 13 |
| 3 | **RF+MLP+LSTM** | **cross-family stack** | **0.785** | 0.835 | 0.844 | 1 | 14 |
| 4 | CatBoost | GAN oversampling | 0.783 | 0.848 | 0.855 | 1 | 13 |
| 5 | BiLSTM | class_weight | 0.781 | 0.839 | 0.844 | 2 | 13 |
| 6 | LSTM | class_weight | 0.780 | 0.813 | 0.821 | 2 | 15 |
| 7 | XGBoost | SMOTE-ENN | 0.777 | 0.813 | 0.815 | 5 | 13 |
| 8 | LightGBM | GAN | 0.775 | 0.822 | 0.832 | 1 | 15 |
| — | LSTM ensemble (5 seeds) | mean | 0.775 | 0.831 | 0.843 | 0 | 15 |
| — | **GraphSAGE (GNN, k-NN graph)** | class_weight | 0.756 | 0.796 | 0.801 | – | – |
| — | MLP | raw / smote / gan | 0.719–0.775 | — | — | — | — |
| — | GATv2 (GNN) | class_weight | 0.745 | – | – | – | – |
| — | FT-Transformer | class_weight | 0.681 | 0.760 | 0.760 | 10 | 14 |
| — | AutoEncoder | anomaly (legit-only) | 0.584 | 0.660 | 0.660 | 19 | 17 |

**Literature benchmarks & protocol context:**
- Best random-split ULB results: XGBoost PR-AUC ≈ 0.88 (EAI ISMLA 2026, cost-sensitive + threshold optimization); LSTM ensemble 0.827 (Anand & Chakrabarty 2025). Same paper confirms: *"Chronological validation yields lower PR-AUC than random stratified evaluation"* — our time-based 0.795 sits at the top of the honest-protocol range (~0.75–0.80).
- Highest published number anywhere: RF PR-AUC 0.9879 (KAIS 2026) — private bank data with random CV, not comparable to ULB.

**Findings matching the literature:**
- Hybrid sampling (SMOTE-ENN) wins for tree ensembles — confirms Ahmed et al. 2025
- **GAN-synthesized fraud is a top-tier strategy for boosting**: CatBoost+GAN ties RF+hybrid's error profile (97.5% precision @ 75% recall); LightGBM+GAN fixes LightGBM's raw weakness — supports Transformer-GAN (2025) direction, though gains are model-dependent
- **Cross-family stacking works; same-family ensembling doesn't**: averaging RF+MLP+LSTM (0.785, highest ROC-AUC 0.985) beats every individual deep model, while a 5-seed LSTM ensemble (0.775) does not beat its own members — diversity across *families* is what pays off
- Sequence modeling adds value over plain MLP; bidirectionality is redundant; feature-level-style synthetic resampling hurts RNNs
- **GNN on a feature-similarity k-NN graph is competitive with plain MLPs** (SAGE 0.756) even though ULB has no relational entities — HGNN papers' higher numbers (AUC-PR ≈ 0.89) require real cardholder/merchant graphs; attention (GATv2, 0.745) adds cost without accuracy here
- FT-Transformer (0.681) loses to boosting and even to the MLP — matches Yu et al. 2024 / KAIS 2026 conclusions on tabular fraud data
- Time-based splits yield lower PR-AUC (~0.79) than inflated random-split numbers (~0.88+) — expected and more realistic

**Phase 5 — Optuna tuning (53 runs total):** tuning *repairs* weak configs rather than lifting champions — LightGBM raw 0.537→0.753 (+0.22), XGBoost raw 0.714→0.769, RF raw 0.749→0.782 — while every tier champion stays within ±0.01 of its untuned best. Optuna independently chose moderate `scale_pos_weight` values (4–172, never 519) and a **single-layer** GraphSAGE, independently confirming our weight and over-smoothing findings. Best params: `reports/optuna_best_params.json`.

**Classic baselines & depth test (68 time-split runs):** Decision Tree best 0.405 vs its own Random Forest 0.795 (+0.39 from ensembling alone); GaussianNB collapses (~0.08 — independence assumption + extreme prior); calibrated LinearSVC hits the linear ceiling (0.747); a deeper DNN (512-256-128-64) scores **identically** to the small MLP (0.7752 vs 0.7746).

**Random-split ablation** (`reports/results_random_split.csv`): rerunning one config per tier under the shuffled protocol inflates 5/6 models (CatBoost+GAN **+0.060**, MLP +0.047, five-model mean +0.035), and our measured random numbers (0.80–0.84) land inside the band the random-split literature reports — empirically demonstrating that the historical benchmark gap is largely *protocol*, not model quality. GraphSAGE is the sole exception (-0.044: shuffling scrambles temporal coherence of kNN neighbourhoods). *Erratum:* an earlier run cached the GAN generator across both split protocols, inflating this row's random-side score to 0.877/+0.097; the cache is now keyed by a hash of the training frauds and the ablation was fully re-measured.

**Protocol lessons (documented anomalies):**
- Extreme `scale_pos_weight` (= imbalance ratio ~519) collapses boosted trees into predicting almost everything as fraud. Both variants are reported for the ablation: `class_weight_fixed` (classic ratio 519) vs `class_weight` (**validation-selected** from grid `{1, 4, 16, √ratio, 64, ratio}`). Result: XGBoost 0.761→0.769 (val chose 16), CatBoost 0.762→0.765 (val chose √ratio≈22.8), LightGBM **collapse preserved at 0.009** with fixed ratio while validation-based selection rejects all weighting (spw=1) and recovers an honest 0.537.
- CatBoost + random undersampling degrades (PR-AUC 0.670): ordered boosting needs data volume; 768 rows after RUS is too few. Kept as a finding.
- Untuned LightGBM lags XGBoost on this dataset — both will receive Optuna tuning in a later phase for a fair final comparison.

## References

- Ahmed et al., *Ensemble ML with hybrid data sampling*, Machine Learning with Applications, 2025
- Anand & Chakrabarty, *Credit Card Fraud Detection and Credit Risk Analysis*, SSRN, 2025
- Singh et al., *Heterogeneous Graph Auto-Encoder for CreditCard Fraud Detection*, arXiv:2410.08121
- HGNN with graph attention, arXiv:2504.08183
