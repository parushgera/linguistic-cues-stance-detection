# Style Markers of Stance

Code and analysis artifacts for the \*SEM 2026 paper (co-located with ACL 2026):

> **Understanding the Linguistic Cues Behind Stance Detection**
> Parush Gera and Tempestt Neal.
> *Proceedings of \*SEM 2026, co-located with ACL 2026.*

> ⚠️ **Working repository.** This repo is under active cleanup as the paper
> moves through the publication pipeline. Layout, scripts, and reproduction
> instructions may shift between commits until the camera-ready is officially
> released. Once \*SEM 2026 proceedings are out, this notice will be removed
> and a stable release will be tagged.

This is an **analysis paper**, not a model release. The repository ships the
training pipeline that produces six stance-detection models, the 43-feature
stylometric extractor, and the SHAP / Logistic-Regression analysis that builds
Tables 4–10 and Figures 1–4. The paper itself is not redistributed in this
repository — please refer to the official \*SEM 2026 proceedings once published.

---

## TL;DR

We train six neural stance detectors (BiGRU, Att-BiGRU, BiLSTM, Att-BiLSTM,
KimCNN, BERT) on a unified corpus of four benchmarks (SemEval-2016, PStance,
COVID-19 Stance, WT-WT) totalling **61,436 train / 15,646 test** examples. We
keep only the **10,527 consensus samples** that *all six* models agree on
(9,769 correct, 758 misclassified) and ask: *which 43 hand-crafted stylistic
features explain when models succeed and when they fail?* We answer with two
complementary lenses — Logistic-Regression coefficients and SHAP values — over
**13 balanced sub-samples per stance class**, producing the consensus-quartile
rankings (Q1–Q4) that drive the paper's claims.

Headline finding: each stance class has a distinct stylistic profile.

| Stance | Reliable when texts are… | Errors when texts are… |
|---|---|---|
| **favor** | concise, direct, low hedging, low pronoun/noun density | long, hedged, with mixed certainty markers |
| **against** | emotionally charged (anger / fear), lexically varied, content-heavy | neutral, formally structured, low affect |
| **none** | short, simple, lexically uniform, emotionally flat | long, emotionally charged (esp. fear) |

---

## Repository layout

```
style_markers/
├── README.md                     # this file
├── requirements.txt              # pinned Python dependencies
├── baselines/                    # the six model architectures (BiGRU/ATBiGRU/BiLSTM/ATBiLSTM/KimCNN/BERT)
├── configs/
│   └── best_parameters.json      # Optuna-tuned hyperparameters per model × dataset × target
├── dataset/
│   ├── all_combined.csv          # unified train+test across all four datasets (paper input)
│   ├── dataset_summary*.csv      # per-target / per-dataset counts (Table 1)
│   ├── processed/                # test splits with the 43 features extracted
│   ├── semeval2016/              # raw SemEval-2016 Task 6 files
│   └── wtwt/                     # raw WT-WT files
├── lexicons/
│   └── subjclueslen1-HLTEMNLP05.tff  # MPQA Subjectivity Clues (certainty / doubt markers)
├── scripts/
│   ├── train.py                          # train all six models, save consensus indices
│   ├── extract_features.py               # extract the 43 stylistic features
│   ├── model_performance_analysis.py     # 13-iteration LR + XGBoost F1 (§4.2.1)
│   ├── shap_analysis.py                  # 13-iteration LR + SHAP (§4.2.2)
│   ├── quartile_analysis.py              # consensus-quartile directional analysis (§5)
│   ├── feature_label_map.py              # raw-name → display-name mapping for plots
│   ├── mappings.py                       # dataset / target / label maps
│   ├── utils.py                          # data loaders, training/eval helpers
│   ├── fine_tune_utils.py                # transformer fine-tuning helpers
│   ├── train.sh                          # SLURM driver for train.py
│   └── extract_features.sh               # SLURM driver for extract_features.py
├── notebooks/
│   ├── final_analysis_script.ipynb       # interactive version of model_performance_analysis.py
│   ├── importance_analysis.ipynb         # qualitative error analysis (writes results/full_error_analysis_output/)
│   └── bar_plot_sub_plots.ipynb          # builds Figure 4 (cross-stance Q1 features)
├── results/                              # all artifacts cited in the paper
│   ├── model_performance_summary.csv     # Table 2 numbers
│   ├── model_performance_detailed.csv
│   ├── Features_in_2__Conditions.csv     # input to bar_plot_sub_plots.ipynb (Figure 4 source)
│   ├── features_subplot.{png,pdf}        # Figure 4 (cross-stance Q1 features)
│   ├── evaluation_results/               # consensus indices + F1s from train.py
│   ├── iterations/                       # 13 LR-only iterations (top features per stance)
│   ├── final_analysis/                   # confusion matrices + ROC curves (LR + XGBoost)
│   ├── final_analysis_multiple_undersampling/  # 13-iteration F1 / mutual-info breakdowns
│   ├── comprehensive_shap_analysis/      # 13 SHAP iterations + aggregations + LR comparison
│   ├── quartile_analysis_final/          # Tables 4–6 / 8–10, Figures 1–3 (per-stance directional plots)
│   └── full_error_analysis_output/       # qualitative report + per-stance samples
├── docs/
│   └── QUARTILE_ANALYSIS_SUMMARY.md      # design notes for the consensus-quartile method
└── archive/                              # everything **not** used in the camera-ready paper
                                          # (early experiments, GloVe baselines, threshold studies,
                                          #  per-target fine-tuning, old plots/notebooks). Safe to delete.
```

> **Why is there an `archive/` folder?** This repo grew over many iterations
> and many ideas that did not make it into the final paper. Everything inside
> `archive/` was preserved for traceability only and is **not** required to
> reproduce any number, table, or figure in the paper.

---

## Datasets

The paper uses four publicly available stance-detection corpora, summarised in
`dataset/dataset_summary.csv` and Table 1 of the paper:

| Dataset | Domain / Targets | Train | Test |
|---|---|---:|---:|
| SemEval-2016 Task 6 | Atheism, Climate Change, Donald Trump, Feminist Movement, Hillary Clinton, Legalization of Abortion | 3,444 | 1,426 |
| PStance | Bernie Sanders, Donald Trump, Joe Biden | 19,417 | 2,157 |
| COVID-19 Stance | face_masks, fauci, school_closures, stay_at_home_orders | 5,333 | 800 |
| WT-WT | Healthcare + Entertainment domains | 33,242 | 11,081 |
| **Combined** | — | **61,436** | **15,646** |

WT-WT's `support / refute / comment / unrelated` labels are mapped to
`favor / against / none / none` (the latter two are merged) by `train.py`.

The **already-merged corpus** is `dataset/all_combined.csv` — this is the
canonical input the paper consumes. `dataset/semeval2016/` and `dataset/wtwt/`
keep the raw upstream files for reference; PStance and COVID-19 are bundled
directly inside `all_combined.csv` (re-create from upstream if needed).

---

## The six models

All defined in `baselines/`:

| Class | Module | Architecture |
|---|---|---|
| `BiGRU` | `bigru.py` | Bidirectional GRU |
| `ATBiGRU` | `atbigru.py` | BiGRU + attention (Zhou et al., 2017) |
| `BiLSTM` | `bilstm.py` | Bidirectional LSTM |
| `ATBiLSTM` | `atbilstm.py` | BiLSTM + attention (Siddiqua et al., 2019) |
| `KimCNN` | `textcnn.py` | Kim (2014) CNN with multiple filter widths |
| `BERTModel` | `bert.py` | `bert-base-uncased`, fine-tuned with a `[CLS]` head |

Non-transformer models receive 384-d **SBERT** (`all-MiniLM-L6-v2`) token
embeddings — *not* GloVe — followed by sequence padding/truncation to 128
tokens. BERT uses its own contextual embeddings learned during fine-tuning.
Hyperparameters were tuned with Optuna (100 trials per architecture); the
final values live in `configs/best_parameters.json` under each model's
`"whole"` key (i.e., the multi-target setting used in the paper).

Test macro-F1 (Table 2):

| Model | F1 |
|---|---:|
| BiGRU | 0.7803 |
| Att-BiGRU | 0.7936 |
| BiLSTM | 0.8006 |
| Att-BiLSTM | 0.7782 |
| KimCNN | 0.7937 |
| BERT | **0.8258** |

These scores are recomputed inside `results/evaluation_results/f1_results.csv`
when `scripts/train.py` runs.

---

## The 43 stylistic features (Table 3)

Extracted by `scripts/extract_features.py`, grouped into six categories:

| Category | Features |
|---|---|
| **Lexical Richness (8)** | word count, sentence count, avg / std word length, type-token ratio, hapax legomena, stopword ratio, punctuation density |
| **Part-of-Speech (5)** | noun, verb, adjective, adverb, pronoun ratios (Penn-Treebank tags via NLTK) |
| **Syntactic / Structural (5)** | function-word ratio, subordinate-clause count, punctuation counts, sentence-length variability, Flesch-Kincaid readability |
| **Affective Tone (8)** | sentiment polarity & subjectivity (TextBlob); 6 emotion scores (anger, joy, fear, sadness, disgust, surprise) from `j-hartmann/emotion-english-distilroberta-base` |
| **Epistemic & Hedging (11)** | certainty/doubt adverb/verb/adjective counts (MPQA Subjectivity Clues); sentence-level hedge probability from `ChrisLiewJY/BERTweet-Hedge`; token-level C/D/E/I/N counts from `jeniakim/hedgehog` |
| **Uncertainty Ratios (5)** | C/D/E/I/N counts normalised by total hedge tokens |

Count-based features are normalised by total word count to make them
comparable across texts.

---

## Reproducing the paper

All scripts resolve paths relative to the repo root, so you can run them from
anywhere as long as you point Python at the file directly. The recommended
workflow is to invoke them **from the repo root** so any imports of
`baselines.*` resolve correctly.

### 0. Environment

```bash
conda create -n style_markers python=3.10
conda activate style_markers
pip install -r requirements.txt

python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords
python -m spacy download en_core_web_sm
```

The first run will additionally cache three Hugging Face models the feature
extractor depends on (~500 MB total):

* `j-hartmann/emotion-english-distilroberta-base` (emotion scores)
* `ChrisLiewJY/BERTweet-Hedge` (sentence-level hedging probability)
* `jeniakim/hedgehog` (token-level hedge categories C/D/E/I/N)

GPU is recommended for training; CPU is fine for the feature-extraction and
analysis steps.

### 1. Train the six models and dump consensus indices

```bash
python scripts/train.py
```

Outputs:

* `results/evaluation_results/f1_results.csv`
* `results/evaluation_results/{wtwt,except_wtwt}_correctly_classified_indices.npy`
* `results/evaluation_results/{wtwt,except_wtwt}_misclassified_indices.npy`

A pre-computed copy of all four `.npy` index files and the matching
feature-augmented test CSVs is shipped in `results/evaluation_results/`, so
you can skip this step if you only want to regenerate the analysis.

### 2. Extract the 43 stylistic features for the test splits

```bash
python scripts/extract_features.py
```

This reads `dataset/processed/{except_wtwt_test,wtwt_test}.csv` and writes
`*_processed.csv` siblings with 43 extra columns. (The pre-computed
`*_processed*.csv` files are already in `dataset/processed/` and
`results/evaluation_results/`.)

### 3. LR + XGBoost feature-validation F1 (Section 4.2.1)

```bash
python scripts/model_performance_analysis.py
```

Reproduces:

* `results/model_performance_summary.csv` — the LR / XGB mean F1 ± std per stance reported in §4.2.1 (e.g. favor LR = 0.68 ± 0.10, against LR = 0.59 ± 0.11, none LR = 0.77 ± 0.05).
* `results/model_performance_detailed.csv` — per-iteration F1s.

### 4. Full SHAP analysis (Section 4.2.2)

```bash
python scripts/shap_analysis.py --mode full
```

Produces, under `results/comprehensive_shap_analysis/`:

* `01_iteration_shap/` … `13_iteration_shap/` — per-iteration importance + raw SHAP values
* `final_shap_aggregated/` — aggregated importance + directional metrics across iterations
* `feature_value_analysis/` — feature-value ↔ SHAP-sign relationships (used in Tables 4–10)
* `shap_linear_comparison_final/` — Spearman correlations between LR coefficients and SHAP rankings (Figures 1–3)

`--mode compare` re-runs only the LR-vs-SHAP comparison if you already have
the per-iteration SHAP outputs.

### 5. Consensus-quartile directional analysis (Section 5, Tables 4–10, Figures 1–3)

```bash
python scripts/quartile_analysis.py
```

Reads from `results/comprehensive_shap_analysis/` and writes to
`results/quartile_analysis_final/`:

* `directional_analysis_{All_Stances,FAVOR,AGAINST,NONE}_all_43_features.{png,pdf}` — Figures 1–3
* `quartile_features_{stance}.csv` — Q1–Q4 assignments
* `unified_quartile_features_all_stances.csv` — full directional breakdown across stances
* `correlation_summary_all_stances.csv` — LR ↔ SHAP rank correlations
* `analysis_summary_report.txt` — text summary

### 6. Figure 4 — cross-stance Q1 features

Open `notebooks/bar_plot_sub_plots.ipynb` and run all cells. It reads
`results/Features_in_2__Conditions.csv` and writes
`results/features_subplot.{png,pdf}`.

### 7. Qualitative error report

Open `notebooks/importance_analysis.ipynb` and run all cells. It rebuilds
`results/full_error_analysis_output/` with the per-stance qualitative
sample CSVs and `analysis_report.txt`.

---

## Citing

The paper appears in the proceedings of \*SEM 2026 (co-located with ACL 2026).
A canonical BibTeX entry will be added here once the proceedings are
published — until then, please use the placeholder below and update it once
the official entry is available:

```bibtex
@inproceedings{gera-neal-2026-stylemarkers,
  title     = {Understanding the Linguistic Cues Behind Stance Detection},
  author    = {Gera, Parush and Neal, Tempestt},
  booktitle = {Proceedings of *SEM 2026 (co-located with ACL 2026)},
  year      = {2026},
  note      = {Placeholder; final BibTeX entry will be updated when the proceedings are published.}
}
```

If you use the consensus-quartile / 13-iteration directional method, please
cite the paper above. The MPQA Subjectivity Clues lexicon (Wilson et al.,
2005) and the three Hugging Face models linked above carry their own
licences — please check each before redistribution.

---

## Notes on what *is not* in the paper

For honesty and reviewer reproducibility, the directories below were part of
the project's exploration but **did not** make it into the camera-ready
paper. They live in `archive/` and can be deleted at any time.

* `archive/glove_baselines/` — earlier per-target / per-dataset training with
  GloVe embeddings. Replaced in the paper by the unified-corpus + SBERT setup.
* `archive/fine_tuning/`, `archive/fine_tuning_scripts/` — per-target and
  per-dataset transformer fine-tuning. The paper trains a single model per
  architecture on the combined corpus.
* `archive/balanced_analysis/`, `archive/today_analysis/`, `archive/paper_plots*` —
  earlier "confidence-threshold" experiments (0.1–0.9). The final paper uses
  unanimous consensus across all six models with no thresholds.
* `archive/classifications*/`, `archive/evaluation_data*/`, `archive/results_old/`,
  `archive/evluation_results/` — earlier output formats.
* `archive/py_scripts/`, `archive/scripts_old/`, `archive/shell_scripts/`,
  `archive/baselines_old/` — superseded by `scripts/` and `baselines/` at the
  repo root.
* `archive/notebooks_old/` — exploratory notebooks (`analysis_*`, `evaluation_*`,
  `meta_*`, `final_analysis_v3`).
* `archive/train_on_grander_master/extract_features_meghna/` — an earlier feature
  extractor (Meghna's version) superseded by `scripts/extract_features.py`.
* `archive/train_on_grander_master/{final_analysis_with_mi,mutual_info.ipynb,12_split_analysis.ipynb,backup_shap_analysis,shap_analysis_backup.py}` —
  exploratory analyses (mutual information, 12-split, alternate SHAP runs)
  that were not adopted in the camera-ready.
