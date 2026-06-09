# KaizenStat

**[www.kaizenstat.com](https://www.kaizenstat.com/)**

[![PyPI Version](https://img.shields.io/pypi/v/kaizenstat.svg?style=flat-square&color=blue)](https://pypi.org/project/kaizenstat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-760%20passed-brightgreen.svg?style=flat-square)]()
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg?style=flat-square)]()
[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg?style=flat-square)]()

**KaizenStat** is a structured Python framework for Data Health Measurement and ML Model Debugging. It enforces a clean, opinionated pipeline — **Health → Validate → Fix → Train → Debug → Improve** — where every decision is explained, scored, and reproducible.

> **v0.5.3 — Polars-powered `load()`**: One command loads any file or URL (CSV, Excel, Parquet, JSON, Feather) using **Polars (Rust)** under the hood — 10–100× faster than pandas on large files. Automatically shows shape, dtypes, missing values, and a 5-row preview. Polars installs automatically on first use. Also fixes pandas 2.x `StringDtype` compatibility.
>
> **v0.5.0** adds full NLP / text data support (auto-detected, zero API changes), a production-readiness Trust Layer, text self-healing with `auto_improve_text()`, and leakage detection for text data.
>
> **Premium Engine (v0.5.1)**: StackingClassifier ensemble (outperforms voting), 2-stage progressive hyperparameter tuning, ExtraTrees in the model pool, model calibration (Platt scaling), multi-pipeline text benchmark with char n-grams, optional sentence embeddings, failure clustering by subgroup, data-vs-model blame diagnosis, and quantified expected-gain suggestions.
>
> **Test Quality**: 760 tests · 100% pass rate · 100% code coverage across all 18 modules (3,127 statements, 0 missed).

---

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [What Makes KaizenStat Different](#what-makes-kaizenstat-different)
- [Architecture](#architecture)
- [Step-by-Step Pipeline](#step-by-step-pipeline)
  - [Step 1 — fit](#step-1--fit)
  - [Step 2 — health](#step-2--health)
  - [Step 3 — validate](#step-3--validate)
  - [Step 4 — fix](#step-4--fix)
  - [Step 5 — train](#step-5--train)
  - [Step 6 — debug\_model](#step-6--debug_model)
  - [Step 7 — improve](#step-7--improve)
  - [Step 8 — report](#step-8--report)
- [AutoML Engine](#automl-engine)
- [NLP / Text Mode (v0.5.0)](#nlp--text-mode-v050)
- [Reliability & Trust Layer (v0.5.0)](#reliability--trust-layer-v050)
- [Data Health Score](#data-health-score)
- [Fix Engine](#fix-engine)
- [Model Debug Engine](#model-debug-engine)
- [Advanced Methods](#advanced-methods)
- [Result Types](#result-types)
- [Module-Level API](#module-level-api)
- [CLI — All Commands](#cli--all-commands)
- [AI Advisor](#ai-advisor-optional)
- [Plugin API](#plugin-api)
- [Developer Setup](#developer-setup)
- [Test Suite & Coverage](#test-suite--coverage)
- [Backward Compatibility](#backward-compatibility)

---

## Install

```bash
pip install kaizenstat
```

Optional extras:

```bash
pip install "kaizenstat[gpu]"   # XGBoost + LightGBM (recommended for best tabular results)
pip install "kaizenstat[nlp]"   # sentence-transformers for embedding-based text models
pip install "kaizenstat[ai]"    # Anthropic Claude AI advisor
pip install "kaizenstat[all]"   # Everything
```

**Requirements:** Python ≥ 3.8 · scikit-learn ≥ 1.1.0 · scipy ≥ 1.7.0 · rich ≥ 12.0.0

---

## Quick Start

```python
from kaizenstat import DataDoctor

doctor = DataDoctor()
doctor.load("data.csv")             # CSV, Excel, Parquet, JSON, Feather, URL — one command
                                    # Powered by Polars (Rust). Shows shape, dtypes, preview automatically.
doctor.fit(target="churn")          # auto-detects tabular vs text mode

doctor.health()                     # Data Health Score 0–100
doctor.validate()                   # statistical + leakage checks
doctor.fix(safe=True)               # preview then apply safe corrections
doctor.train()                      # benchmark + train best model
doctor.debug_model()                # root-cause failure analysis
doctor.improve()                    # prioritised improvement suggestions
doctor.report()                     # terminal summary + HTML export
```

**Works identically for text data** — no API changes:

```python
doctor = DataDoctor()
doctor.load("reviews.csv")          # has a "text" column + "sentiment" label
doctor.fit(target="sentiment")      # → Mode: TEXT ('text')

doctor.health()                     # text quality: noise, duplicates, vocabulary, imbalance
doctor.validate()                   # token skew, stopword dominance, label leakage
doctor.train()                      # TF-IDF + size-adaptive classifier (auto-chosen)
doctor.debug_model()                # sparse matrix, rare-token overfitting, bias slices
doctor.improve()                    # n-grams, char n-grams, embeddings, augmentation
```

---

## What Makes KaizenStat Different

```
❌ Others:    "Find best model → Accuracy: 1.0"
✅ KaizenStat: "This result is fake. Your data is leaking."
```

| Capability | Description |
|---|---|
| **Leakage Detection** | Flags features with corr > 0.98 to target — prints `🚨 Leakage detected` and prevents blind training |
| **Text Label Leakage** | Detects giveaway tokens with ≥ 98% class concentration in text data (v0.5.0) |
| **Data Intelligence Profile** | Measures imbalance, dimensionality, sparsity, missing ratio before choosing models |
| **Smart Model Selection** | Profile-aware: skips slow models on high-dim data, adds `class_weight='balanced'` for imbalanced targets |
| **AutoML Ensemble** | Soft-voting ensemble of top 3 models — not just the single winner |
| **NLP Auto-Routing** | Detects dominant text column; routes entire pipeline to NLP modules automatically (v0.5.0) |
| **Root Cause AI** | `why_bullets` explains in plain English why the model is behaving the way it is |
| **Counterfactual Impact** | `feature_impact()` measures how much accuracy drops when each feature is removed |
| **Drift Detection** | KS test across train vs test — catches distribution shift before it hits production |
| **Trust Score** | 0–100 production-readiness score: confidence + robustness + calibration + failure slices (v0.5.0) |
| **Text Self-Healing** | `auto_improve_text()`: baseline → debug → clean → retrain → before-vs-after compare (v0.5.0) |
| **Dataset Difficulty** | Baseline LR cross-val score → 0–1 difficulty rating (Easy / Moderate / Hard) |
| **Recommendation Engine** | `recommend_actions()` returns a ranked, emoji-prefixed what-to-do list |
| **Stacking Ensemble** | `StackingClassifier` with LogReg meta-learner — outperforms soft voting on most real datasets |
| **Progressive Tuning** | 2-stage coarse→fine hyperparameter search — beats single-pass random search at same budget |
| **Failure Clustering** | Automatically finds which subgroups fail most (`city='NY': 52% vs 81% overall`) |
| **Data vs Model Blame** | Fits a RandomForest baseline to decide if the problem is bad data or a weak model |
| **Model Calibration** | Platt scaling auto-applied when model is overconfident (confidence − accuracy > 0.10) |
| **Text Multi-Benchmark** | Races word n-gram, char n-gram, and LinearSVC pipelines — picks the winner automatically |
| **Sentence Embeddings** | Upgrades to `all-MiniLM-L6-v2` embeddings when they outperform TF-IDF by > 1% CV |

---

## Premium Engine (v0.5.1)

These upgrades run automatically inside the existing API — no new methods to call.

### Stacking Ensemble (replaces soft voting)

`train_auto(ensemble=True)` now builds a `StackingClassifier` with a LogReg meta-learner instead of a simple soft-voting average. The meta-learner learns the optimal combination of base model predictions via out-of-fold cross-validation — the same approach used in top Kaggle competition solutions.

```python
result = doctor.train_auto(tune=True, ensemble=True)
# → "Stack(LightGBM+XGBoost+ExtraTreesClassifier)"
```

### 2-Stage Progressive Tuning

`train(tune=True)` and `train_auto(tune=True)` now run a two-stage search:

```
Stage 1 — Coarse:  n_iter // 2 iterations over the full param grid
Stage 2 — Fine:    remaining iterations over a ±1-neighbour grid around the best Stage 1 params
```

This reliably beats a single-pass random search of the same total iteration budget and is printed in the terminal:

```
Progressive tuning — Stage 1/2: coarse search (n_iter=10)…
Progressive tuning — Stage 2/2: refining (n_iter=10)…
✓ Progressive tuning — coarse: 0.8421 → fine: 0.8573 (+0.0152)
```

### Automatic Model Calibration (Platt Scaling)

After training, KaizenStat checks whether the final model is overconfident. If `mean(confidence) − accuracy > 0.10`, it automatically wraps the model with `CalibratedClassifierCV(method="sigmoid")`:

```
Applied Platt calibration — model was overconfident
```

This improves ROC AUC and `trust_score()` without changing predictions.

### Extended Model Pool

The benchmark now includes **ExtraTrees** alongside Random Forest, Gradient Boosting, XGBoost, and LightGBM. ExtraTrees is fast, highly diverse from Random Forest (no bootstrap, random split thresholds), and an excellent stacking base model.

### Data vs Model Blame

`debug_model()` now runs a RandomForest baseline on your training data to decide whether the problem is in the **data** or the **model**:

| Diagnosis | Condition | Action |
|---|---|---|
| **Data Problem** | Baseline RF also scores < 0.60 CV | Collect better features or fix labels — swapping models won't help |
| **Model Problem** | Baseline RF scores ≥ 0.70 but current model scores much lower | Run `train_auto(tune=True, ensemble=True)` |

### Failure Clustering by Subgroup

`debug_model()` automatically finds WHERE the model fails by checking accuracy per categorical subgroup. Any subgroup more than 15 percentage points below overall accuracy is flagged:

```
Failure Slice: city
Overall 81% but city='NY': 52% (34 samples)  city='Chicago': 65% (28 samples)
```

### Text Multi-Pipeline Benchmark

`train()` in text mode no longer picks a fixed pipeline based on row count. It races all viable candidates and returns the winner:

| Pipeline | Strength |
|---|---|
| `TFIDF+LogReg` | Fast baseline, works on any size |
| `TFIDF_char+LogReg` | Robust for noisy/short text, typos, multilingual |
| `TFIDF+LinearSVC` | Best linear accuracy on medium+ datasets |
| `Embeddings+LogReg` | Semantic understanding via `all-MiniLM-L6-v2` (if installed) |

The benchmark table is printed and the winner is used automatically. Sentence embeddings are tried last and only adopted if they beat TF-IDF by more than 1% CV score.

### Quantified Expected Gains in Suggestions

`improve()` now gives data-driven gain estimates, not generic text:

```
[HIGH]  Ensemble / AutoML    → Run train_auto(tune=True, ensemble=True)
                               Expected: +8–14% accuracy gain from stacking + tuning
[HIGH]  Class Imbalance      → Apply SMOTE or class_weight='balanced'
                               Expected: +20–30% minority-class recall and F1
[MEDIUM] Calibration         → Check trust_score() — apply Platt scaling if gap > 0.05
                               Expected: +0.03–0.08 calibration gap reduction
[HIGH]  Subgroup Fix: city   → Collect more samples for failing subgroup 'city'
                               Expected: +3–8% overall F1
```

---

## Architecture

```
kaizenstat/
├── __init__.py                      # Public API + v0.2 backward compat
│
├── doctor/
│   └── data_doctor.py               # DataDoctor orchestrator — all pipeline methods
│
├── health/
│   ├── scorer.py                    # Tabular: 0–100 Data Health Score with 8 penalties
│   └── text_scorer.py               # Text: empty docs, noise, vocab, imbalance (v0.5.0)
│
├── validate/
│   ├── checker.py                   # Normality, VIF, leakage, skewness, drift (KS test)
│   └── text_checker.py              # Token skew, stopwords, hapax explosion, text leakage (v0.5.0)
│
├── fix/
│   └── engine.py                    # Preview-first FixPlan — safe, typed corrections
│
├── model/
│   ├── trainer.py                   # Benchmark + train_best + train_auto (AutoML engine)
│   └── text_trainer.py              # Size-adaptive TF-IDF pipelines (v0.5.0)
│
├── debug/
│   ├── debugger.py                  # Priority-based diagnosis + feature_impact + recommend_actions
│   └── text_debugger.py             # Sparsity, rare tokens, imbalance bias, top-token coefs (v0.5.0)
│
├── improve/
│   ├── suggester.py                 # Rule-based improvement suggestions (tabular)
│   └── text_suggester.py            # n-grams, embeddings, augmentation, balancing (v0.5.0)
│
├── reliability/
│   ├── __init__.py
│   └── trust.py                     # TrustAnalyzer + TrustReport (v0.5.0)
│
├── intelligence/
│   └── ai_advisor.py                # Optional Anthropic Claude integration
│
├── output/
│   └── reporter.py                  # HTML report, model export, codegen
│
├── cli/
│   └── main.py                      # kz CLI (Typer) — all 9 commands
│
└── utils/
    └── helpers.py                   # Shared utilities + text detection helpers
```

---

## Step-by-Step Pipeline

### Step 1 — `fit`

Register a dataset. Auto-detects whether the data is tabular or text-dominant.

```python
from kaizenstat import DataDoctor
import pandas as pd

doctor = DataDoctor()
doctor.fit(df, target="churn")
# Output:
# ╭── DataDoctor.fit ───────────────────────────────────────────╮
# │ Dataset registered  │  5,000 rows × 12 columns  │  Task: classification  │  Mode: TABULAR
# ╰─────────────────────────────────────────────────────────────╯
```

```python
# Check detected mode
print(doctor.mode())    # "tabular" or "text"
```

**Text mode** is activated automatically when a column has average word count > 3 and average character length > 20 (and is not low-cardinality like a categorical):

```python
doctor.fit(reviews_df, target="sentiment")
# │  Mode: TEXT ('review_text')
```

---

### Step 2 — `health`

Scores the dataset **0–100** across quality dimensions. Prints a penalty breakdown and grade.

```python
result = doctor.health()

result.score        # → 71.0
result.grade        # → "C"
result.risk_level   # → "MEDIUM"
result.penalties    # → list of HealthPenalty objects
result.summary      # → human-readable string
result.display()    # rich terminal panel (called automatically)
```

**Tabular penalties** (up to −20 each, see [Data Health Score](#data-health-score) table below).

**Text-mode penalties** (v0.5.0):

| Penalty | Trigger |
|---|---|
| Empty / very short docs | Rows with ≤ 2 words |
| Near-duplicate documents | Normalised exact-match duplicates |
| Noise | URL ratio, HTML tag ratio, special-char density |
| Vocabulary diversity | Type-token ratio, total vocabulary size |
| Length variance | High coefficient of variation in word counts |
| Class imbalance | Minority class < 10% |

---

### Step 3 — `validate`

Checks statistical assumptions and data integrity. Runs any registered custom checks too.

```python
report = doctor.validate()

report.passed       # → True / False
report.issues       # → list of ValidationIssue objects
report.checks_run   # → int (number of checks executed)
```

**Tabular checks:**
- Normality (Shapiro-Wilk per numeric column)
- Multicollinearity (VIF — Variance Inflation Factor)
- Skewness (|skew| > 3 flagged)
- Feature–target leakage (`🚨 Leakage detected in: [col1, col2]` with plain-English explanation)

**Text-mode checks** (v0.5.0):
- **Token frequency skew** — top-10 tokens account for > 50% of all token occurrences
- **Stopword dominance** — stopword ratio > 55% (built-in list, no NLTK required)
- **Rare-token explosion** — hapax ratio > 60% with vocabulary ≥ 100 (overfitting risk)
- **Text label leakage** — tokens with ≥ 98% concentration in one class + ≥ 2% frequency → prints `🚨 Leakage detected`

**Drift detection** (separate method, works on any split):

```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=["churn"])
X_train, X_test = train_test_split(X, test_size=0.2)

drifted = doctor.detect_drift(X_train, X_test)
# → {"income": 0.0021, "age": 0.041}   (p < 0.05 = significant drift)
```

---

### Step 4 — `fix`

Plans and applies safe data corrections. **Never modifies the original DataFrame silently.**

```python
# Preview only — shows the fix plan table without touching data
doctor.fix(safe=True, preview_only=True)

# Apply safe (LOW-risk) fixes → returns new fixed DataFrame
fixed_df = doctor.fix(safe=True)

# Apply all fixes including MEDIUM-risk
fixed_df = doctor.fix(safe=False)
```

Using the module API directly:

```python
from kaizenstat import fix

plan = fix.plan(df, target="churn", safe=True)   # shows plan table
fixed_df = plan.apply(df)                         # returns new DataFrame
```

**Targeted fix methods:**

```python
fix.missing(df, target="y")          # null-filling only
fix.outlier_handling(df, target="y") # outlier clipping only
fix.encoding(df, target="y")         # label encoding only
fix.imbalance(df, target="y")        # check + advise on class imbalance
```

See the full [Fix Engine table](#fix-engine) below for all 11 fix types.

---

### Step 5 — `train`

Benchmarks all candidate models with cross-validation, then trains the best on a clean train/test split.

```python
result = doctor.train(
    cv=5,           # cross-validation folds (default: 5)
    test_size=0.2,  # held-out test fraction (default: 0.2)
    tune=False,     # RandomizedSearchCV on the winner (default: False)
    n_iter=20,      # hyperparameter combinations when tune=True (default: 20)
)

result.model_name    # → "LightGBM"
result.train_score   # → 0.9421
result.test_score    # → 0.8871
result.cv_score      # → 0.8734  (mean CV score from benchmark)
result.cv_std        # → 0.0121
result.best_params   # → {"model__n_estimators": 200, ...}  (populated when tune=True)
result.task          # → "classification"
result.pipeline      # → sklearn Pipeline (ready for inference)
```

**With hyperparameter tuning:**

```python
result = doctor.train(cv=5, tune=True, n_iter=20)
# RandomizedSearchCV on the benchmark winner — ~3–5 min on typical datasets
```

**Full AutoML in one call** (see [AutoML Engine](#automl-engine)):

```python
result = doctor.train_auto(cv=3, tune=True, ensemble=True)
# → "Stack(LightGBM+XGBoost+ExtraTreesClassifier)"
```

**Text mode** — `train()` benchmarks all viable pipelines and picks the winner automatically:

| Pipeline | Best for |
|---|---|
| `TFIDF + LogReg` (word 1–2 grams) | Fast baseline, any dataset size |
| `TFIDF_char + LogReg` (char 3–5 grams) | Noisy text, typos, short docs, multilingual |
| `TFIDF + LinearSVC` (calibrated) | Highest accuracy on medium–large datasets (n ≥ 500) |
| `Embeddings + LogReg` (`all-MiniLM-L6-v2`) | Semantic understanding — adopted only if +1% CV gain |

The benchmark table is printed and the winner is trained automatically. All pipelines support `tune=True`.

---

### Step 6 — `debug_model`

Diagnoses why the model is failing. Runs `train()` automatically if not done yet.

```python
result = doctor.debug_model()

result.label               # → "overfitting"
result.severity            # → "HIGH"
result.confidence          # → 0.85
result.health_score        # → 62 / 100
result.gap                 # → 0.143  (train_score − test_score)
result.avg_score           # → 0.874
result.diagnosis           # → "Model generalises poorly to unseen data"
result.root_cause          # → "High variance — likely too many features or tree depth"
result.why_bullets         # → ["Train score 0.94 vs test 0.80 — 14% gap", ...]
result.feature_importances # → pd.Series (feature → importance, sorted desc)
result.issues              # → list of DebugIssue (failure slices, data vs model blame)
```

**Premium diagnostics** run automatically inside `debug_model()`:
- **Failure Clustering** — checks per-categorical-subgroup accuracy; any group > 15pp below overall is flagged (`Failure Slice: city — 'NY': 52% vs 81% overall`)
- **Data vs Model Blame** — fits a RandomForest baseline on numeric features; if baseline < 0.60 → Data Problem; if baseline ≥ 0.70 but current model is much lower → Model Problem

**Text-mode debug** (v0.5.0) additionally reports:
- Sparse matrix statistics (sparsity %, vocabulary size, avg non-zero tokens)
- Weak representation — many OOV tokens on test set
- Rare-token overfitting — hapax ratio > 50%
- Class-imbalance prediction bias — per-class recall imbalance
- Top predictive tokens per class (from TF-IDF coefficients)

See the full [Model Debug Engine table](#model-debug-engine) for all 13 diagnostic labels.

---

### Step 7 — `improve`

Generates a prioritised, ranked list of what to fix next.

```python
report = doctor.improve()

report.suggestions    # → list of Suggestion objects
report.top_priority   # → highest-priority Suggestion

for s in report.suggestions:
    print(f"[{s.impact}] {s.action}  — {s.expected_gain}")
```

**Tabular suggestions** include quantified gain estimates derived from your actual metrics:

| Category | Example gain estimate |
|---|---|
| Data Volume | "+5–15% test score improvement per 2× data increase" |
| Class Imbalance | "+20–30% minority-class recall and F1" (derived from actual imbalance ratio) |
| Ensemble / AutoML | "+8–14% accuracy gain from stacking + tuning" (gap-to-0.80 based) |
| Model Tuning | "+3–10% accuracy gain (progressive search beats single random search)" |
| Calibration | "+0.03–0.08 calibration gap reduction" |
| Subgroup Fix | "+3–8% overall F1" (emitted when failure clustering finds a failing slice) |
| Feature Selection | "+1–3% generalization gain" |

**Text-mode suggestions** (v0.5.0):
- Char n-grams when sparsity is high
- Bigrams/trigrams when representation is weak
- `min_df` tuning to prune rare tokens
- Class-weighting / SMOTE for imbalanced labels
- Embedding-based models when TF-IDF test score is low
- Data augmentation for small datasets (< 1,000 rows)
- Hyperparameter tuning call

---

### Step 8 — `report`

Prints a terminal summary and exports a full HTML report.

```python
path = doctor.report(
    output_path="report.html",   # default: "kaizenstat_report.html"
    open_browser=True,           # auto-open in browser (default: False)
)
# → "report.html"
```

**Export and codegen:**

```python
# Save trained pipeline to disk
doctor.export_model(path="model.joblib")

# Generate standalone Python script — no KaizenStat dependency in production
doctor.codegen(output_path="pipeline.py")
```

---

## AutoML Engine

`train_auto()` runs a 5-step data-intelligence pipeline:

```
Step 1 · Build data profile   → n_rows, n_cols, imbalance ratio, missing %, dimensionality, sparsity
Step 2 · Smart model set       → skips slow models on high-dim data; adds class_weight='balanced' for imbalance
Step 3 · Feature selection     → SelectKBest(top 50) when n_features > 50
Step 4 · Benchmark + tune      → CV benchmark on train set; 2-stage progressive RandomizedSearchCV on winner
Step 5 · Stacking ensemble     → StackingClassifier (LogReg meta-learner, cv=3) / StackingRegressor (Ridge)
Step 6 · Calibration check     → auto-applies Platt scaling if confidence − accuracy > 0.10
```

```python
result = doctor.train_auto(
    cv=3,           # cross-validation folds (default: 3)
    test_size=0.2,  # held-out test fraction
    tune=True,      # 2-stage progressive hyperparameter search on winner
    n_iter=20,      # total hyperparameter combinations (split across 2 stages)
    ensemble=True,  # StackingClassifier / StackingRegressor from top models
)
# result.model_name → "Stack(LightGBM+XGBoost+ExtraTreesClassifier)"
```

**Direct data profile access:**

```python
from kaizenstat.model.trainer import ModelTrainer

X = df.drop(columns=["target"])
y = df["target"]
profile = ModelTrainer._analyze_data(X, y)
# → {
#     "n_rows": 5000, "n_cols": 18,
#     "imbalance": 0.08,       # minority class share
#     "high_dim": False,       # n_features > 50
#     "missing_ratio": 0.03,
#     "sparse": False,
#   }
```

---

## NLP / Text Mode (v0.5.0)

Text mode is **automatic** — no new methods, no flags to set. When `fit()` detects a dominant text column (avg word count > 3.0, avg char length > 20.0, not low-cardinality), all pipeline methods route to NLP-specific modules.

### Mode detection

```python
doctor.fit(df, target="sentiment")
print(doctor.mode())   # "text" or "tabular"
```

### Full text pipeline

```python
# Exact same calls as tabular — all internally routed to NLP modules
doctor.health()          # TextHealthScorer — 6 text-quality penalties
doctor.validate()        # TextValidator — token skew, stopwords, leakage
doctor.train()           # TextModelTrainer — TF-IDF + size-adaptive classifier
doctor.debug_model()     # TextModelDebugger — sparsity, rare tokens, bias slices
doctor.improve()         # TextSuggester — n-grams, embeddings, augmentation
```

### Text self-healing loop

```python
# Baseline → debug → clean noise/URLs/HTML → retrain → compare
comparison = doctor.auto_improve_text(tune=True)

comparison.score_delta    # → +0.031
comparison.display()      # Before vs After panel (called automatically)
```

The healer applies: URL removal, HTML tag stripping, whitespace normalization, empty-document pruning.

### Text-specific module API

```python
from kaizenstat.health import text_scorer
from kaizenstat.validate import text_checker
from kaizenstat.model import text_trainer
from kaizenstat.debug import text_debugger
from kaizenstat.improve import text_suggester

text_scorer.report(df, target="label", text_col="text")
text_checker.assumptions(df, target="label", text_col="text")
text_trainer.train_best(df, target="label", text_col="text", tune=True)
text_debugger.model_failure(pipeline, X_train, X_test, y_train, y_test)
text_suggester.suggest(df, target="label", text_col="text",
                       health_result=hr, debug_result=dr)
```

---

## Reliability & Trust Layer (v0.5.0)

`trust_score()` answers: **can I trust this model in production?**

```python
# Requires train() or train_auto() first; reuses debug_model() split if available
report = doctor.trust_score()

report.trust_score         # → 74  (0–100)
report.grade               # → "needs review"  / "production-ready" / "not ready"
report.confidence_mean     # → 0.81
report.confidence_std      # → 0.14
report.uncertain_fraction  # → 0.17  (17% of predictions are low-confidence)
report.robustness_score    # → 0.89  (prediction agreement under input perturbation)
report.calibration_gap     # → 0.063 (|mean confidence − accuracy|, lower = better)
report.failure_slices      # → ["city='NY': accuracy 59%", "Low-confidence band: accuracy 45%"]
report.notes               # → ["Calibration gap is above 0.05 — consider Platt scaling"]
```

**Trust score formula:**

```
trust_score = 0.40 × accuracy
            + 0.25 × robustness
            + 0.20 × (1 − calibration_gap)
            + 0.15 × (1 − uncertain_fraction)
```

| Grade | Score | Meaning |
|---|---|---|
| production-ready | ≥ 80 | Safe to deploy |
| needs review | 60–79 | Address failure slices first |
| not ready | < 60 | Do not deploy — reliability too low |

**Works in both tabular and text mode** — perturbation adapts automatically:
- **Tabular:** Gaussian jitter on numeric features
- **Text:** Random word-dropout perturbation

**Failure slicing** reports:
- Low-confidence band accuracy (predictions with confidence < 0.60)
- Per-class recall (detects class-specific prediction bias)
- Categorical subgroup accuracy (for each categorical column in tabular mode)

---

## Data Health Score

Scores your dataset **0–100** across 8 penalty categories (tabular mode):

| Penalty | Max Deduction | Trigger |
|---|---|---|
| Missing Values | −20 | Any column with NaN |
| Duplicate Rows | −10 | Exact row duplicates |
| Class Imbalance | −20 | Minority class < 10% |
| Outliers | −10 | > 1% of rows beyond 3×IQR |
| High Skewness | −10 | \|skew\| > 3 |
| Constant Features | −5 | Zero-variance columns |
| High Cardinality | −8 | Categorical column > 50 unique values |
| Leakage Proxy | −20 | Feature correlation > 0.98 with target |

**Grades:** A (≥ 90) · B (≥ 80) · C (≥ 70) · D (≥ 60) · F (< 60)

Module-level access:

```python
from kaizenstat import health

score = health.score(df, target="churn")          # → float (0–100)
result = health.report(df, target="churn")        # → HealthResult (with .display())
result = health.breakdown(df, target="churn")     # → HealthResult with full penalty list
```

---

## Fix Engine

The fix engine **never modifies data silently**. Every fix is planned first, shown as a table, and only applied on `.apply()`.

```python
# 1. Preview — show every planned action with risk level and reason
doctor.fix(safe=True, preview_only=True)

# 2. Apply — returns a NEW DataFrame; original df is untouched
fixed_df = doctor.fix(safe=True)

# 3. Apply all fixes including MEDIUM-risk
fixed_df = doctor.fix(safe=False)
```

`safe=True` (default) restricts to `LOW`-risk actions only.

### What the Fix Engine detects and heals

| Problem Detected | Action Applied | Risk |
|---|---|---|
| Duplicate rows | Drop exact duplicates | LOW |
| Missing target rows | Drop rows where target is null | LOW |
| Constant / zero-variance columns | Drop the column | LOW |
| ID-like columns (unique per row) | Drop the column (leakage risk) | LOW |
| Categorical columns | Label encode to numeric | LOW |
| Numeric nulls — < 50% missing | Fill with **median** | LOW |
| Categorical nulls — < 50% missing | Fill with **mode** (most frequent value) | LOW |
| Numeric nulls — > 50% missing | Drop the column (too unreliable) | MEDIUM |
| Categorical nulls — > 50% missing | Drop the column (too unreliable) | MEDIUM |
| Extreme outliers (> 3×IQR, affects > 2% rows) | Clip to [1%, 99%] percentile range | MEDIUM |
| Skewed numeric features (\|skew\| > 2, min ≥ 0) | Apply `log1p` transform | MEDIUM |

---

## Model Debug Engine

Uses a **priority-based classifier** to diagnose model performance from `train_score` and `test_score`.

| Label | Condition | Severity | Confidence |
|---|---|---|---|
| `data_leakage` | train = 1.0 AND test = 1.0 | CRITICAL | 0.99 |
| `leakage_risk` | both ≥ 0.98 | HIGH | 0.95 |
| `data_issue` | test > train | CRITICAL | 0.98 |
| `severe_underfitting` | both ≤ 0.60 | CRITICAL | 0.95 |
| `underfitting` | both ≤ 0.70 | HIGH | 0.90 |
| `excellent` | gap ≤ 0.05 AND test ≥ 0.90 | LOW | 0.95 |
| `healthy` | gap ≤ 0.05 AND test ≥ 0.80 | LOW | 0.90 |
| `acceptable` | gap ≤ 0.05 | LOW | 0.80 |
| `overfitting_risk` | 0.05 < gap ≤ 0.10 | MEDIUM | 0.75 |
| `overfitting` | 0.10 < gap ≤ 0.20 | HIGH | 0.85 |
| `severe_overfitting` | gap > 0.20 | CRITICAL | 0.95 |
| `weak_model` | gap > 0.15 AND test < 0.70 *(override)* | HIGH | 0.90 |
| `broken_model` | gap > 0.30 AND test < 0.60 *(override)* | CRITICAL | 0.98 |

Each `DebugResult` includes `label`, `severity`, `confidence`, `health_score` (0–100), `gap`, `avg_score`, `diagnosis`, `root_cause`, `why_bullets` (plain-English root cause AI), and `feature_importances` (sorted `pd.Series`).

---

## Advanced Methods

### Counterfactual Feature Impact

Drop each feature one at a time and measure how much the score drops.

```python
doctor.train()

impacts = doctor.feature_impact(top_n=15)
# → {"credit_score": 0.124, "age": 0.031, "city": -0.002, ...}
# Negative = that feature was hurting the model
```

### Dataset Difficulty

Estimate how hard the dataset is using a baseline linear model.

```python
difficulty = doctor.dataset_difficulty()
# → 0.38   (Moderate — baseline LR gets ~62% accuracy)
# → 0.0    (Easy — trivially separable)
# → 0.9    (Hard — near-random performance)
```

### Recommendation Engine

Get a prioritised action list based on data profile + debug result.

```python
doctor.train()
doctor.debug_model()

actions = doctor.recommend_actions()
# → [
#     "⚡ Class imbalance (8% minority) → apply SMOTE (+10–20% F1 expected)",
#     "🔧 Test score 0.64 → run train_auto(tune=True) (+5–15% expected)",
#     "🗑️  3 near-zero importance features → drop them",
#   ]
```

### Auto Improve (Tabular)

Apply safe data fixes then retrain and compare.

```python
comparison = doctor.auto_improve(tune=True)

comparison.before.test_score   # → 0.813
comparison.after.test_score    # → 0.856
comparison.score_delta         # → +0.043
comparison.display()           # Before vs After panel (called automatically)
```

### Pipeline Confidence Score

Overall pipeline health rolled into a single 0–100 score.

```python
doctor.health()
doctor.validate()
doctor.train()
doctor.debug_model()

confidence = doctor.pipeline_confidence()
# → 74  ("needs work")
```

---

## Result Types

| Class | Key Fields |
|---|---|
| `HealthResult` | `score`, `grade`, `risk_level`, `penalties`, `summary` |
| `ValidationReport` | `passed`, `issues`, `checks_run` |
| `FixPlan` | `actions`, `safe`; `.apply(df) → DataFrame` |
| `TrainResult` | `model_name`, `task`, `train_score`, `test_score`, `cv_score`, `cv_std`, `best_params`, `metrics`, `pipeline` |
| `BenchmarkResult` | `task`, `metric`, `entries`, `best_name`, `best_score`, `best_pipeline` |
| `DebugResult` | `label`, `severity`, `confidence`, `health_score`, `gap`, `avg_score`, `diagnosis`, `root_cause`, `why_bullets`, `feature_importances`, `issues` (failure slices + blame) |
| `ImprovementReport` | `suggestions`, `top_priority` |
| `TrustReport` | `trust_score`, `grade`, `confidence_mean`, `confidence_std`, `uncertain_fraction`, `robustness_score`, `calibration_gap`, `failure_slices`, `notes` |
| `ComparisonResult` | `before`, `after` (both `TrainResult`); `.score_delta` property |

All result objects have a `.display()` method for rich terminal output.

---

## Module-Level API

Every capability is available without `DataDoctor` via module-level functions:

```python
from kaizenstat import health, validate, fix, model, debug, improve, reliability

# Health
health.score(df, target="y")                              # → float 0–100
health.report(df, target="y")                             # → HealthResult
health.breakdown(df, target="y")                          # → HealthResult with full penalty list

# Validate
validate.assumptions(df, target="y")                      # → ValidationReport
validate.leakage(df, target="y")                          # → ValidationReport

# Drift (requires two DataFrames)
from kaizenstat.validate.checker import detect_drift
detect_drift(X_train, X_test)                             # → {col: p_value}

# Fix
from kaizenstat import fix
plan = fix.plan(df, target="y", safe=True)                # → FixPlan (shows table)
fixed_df = plan.apply(df)                                 # → new DataFrame

# Model
model.benchmark(df, target="y")                           # → BenchmarkResult
model.train_best(df, target="y", tune=True)               # → TrainResult
model.train_auto(df, target="y", ensemble=True)           # → TrainResult (AutoML)
model.evaluate(pipeline, X_test, y_test)                  # → dict of metrics

# Debug
debug.model_failure(pipe, X_tr, X_te, y_tr, y_te)        # → DebugResult

from kaizenstat.debug.debugger import (
    feature_impact, dataset_difficulty, recommend_actions
)
feature_impact(pipe, X_test, y_test)                      # → {feature: score_drop}
dataset_difficulty(X, y)                                  # → float 0–1
recommend_actions(profile, debug_result)                  # → list of strings

# Improve
improve.suggest(df, target="y",
    health_result=hr, debug_result=dr)                    # → ImprovementReport
improve.prioritize(suggestions)                           # → sorted list

# Reliability (v0.5.0)
from kaizenstat.reliability.trust import TrustAnalyzer
analyzer = TrustAnalyzer()
report = analyzer.analyze(pipeline, X_test, y_test, task="classification")
```

**Text module APIs (v0.5.0):**

```python
from kaizenstat.health.text_scorer import TextHealthScorer
from kaizenstat.validate.text_checker import TextValidator
from kaizenstat.model.text_trainer import TextModelTrainer
from kaizenstat.debug.text_debugger import TextModelDebugger
from kaizenstat.improve.text_suggester import TextSuggester

# All follow the same interface as their tabular counterparts
TextHealthScorer().report(df, target="label", text_col="text")
TextValidator().assumptions(df, target="label", text_col="text")
TextModelTrainer().train_best(df, target="label", text_col="text", tune=True)
TextModelDebugger().model_failure(pipeline, X_train, X_test, y_train, y_test)
TextSuggester().suggest(df, target="label", text_col="text",
                        health_result=hr, debug_result=dr)
```

---

## CLI — All Commands

Install KaizenStat and the `kz` command is available immediately.

```bash
pip install kaizenstat
kz --help
```

### Data Commands

**`kz health`** — Compute and display the Data Health Score.

```bash
kz health data.csv --target churn
kz health data.csv -t churn
```

**`kz validate`** — Run statistical assumption and leakage checks.

```bash
kz validate data.csv --target churn
# Prints 🚨 Leakage detected in: [col1, col2] when feature corr > 0.98
```

**`kz fix`** — Preview or apply safe data corrections.

```bash
kz fix data.csv --target churn --preview          # show plan, do not apply
kz fix data.csv --target churn -o data_fixed.csv  # apply + save to file
kz fix data.csv --target churn                    # apply, auto-name output file
```

**`kz improve`** — Prioritised improvement suggestions.

```bash
kz improve data.csv --target churn
```

### Model Commands

**`kz train`** — Benchmark models and train the best one.

```bash
kz train data.csv churn
kz train data.csv churn --cv 5
kz train data.csv churn --tune                        # RandomizedSearchCV on winner
kz train data.csv churn --tune --n-iter 30            # more combinations (default: 20)
kz train data.csv churn --export model.joblib         # save pipeline to disk
kz train data.csv churn --tune --export model.joblib  # tune + save
```

**`kz debug`** — Full model debug analysis.

```bash
kz debug data.csv churn
# Runs train() internally then prints DebugResult with label, severity, why_bullets
```

**`kz export`** — Train the best model and save it to a `.joblib` file.

```bash
kz export data.csv churn -o model.joblib
kz export data.csv churn --cv 5 -o model.joblib
```

**`kz codegen`** — Generate a standalone Python training script.

```bash
kz codegen data.csv churn -o pipeline.py
# Outputs a self-contained script with no KaizenStat dependency
```

### Report Commands

**`kz report`** — Generate a full HTML pipeline report.

```bash
kz report data.csv --target churn
kz report data.csv --target churn -o my_report.html
kz report data.csv --target churn -o report.html --open   # auto-open in browser
```

### Full Pipeline

**`kz auto`** — Run the complete pipeline in a single command.

```bash
kz auto data.csv churn
kz auto data.csv churn -o report.html

# Internally runs: health → validate → fix → train → debug → improve → report
```

### Command Reference

| Command | Required | Options | What it does |
|---|---|---|---|
| `kz health` | `file` | `--target / -t` | Data Health Score 0–100 with penalty breakdown |
| `kz validate` | `file` | `--target / -t` | Statistical + leakage checks; prints `🚨` on leakage |
| `kz fix` | `file` | `--target / -t`, `--preview / -p`, `--output / -o` | Preview and apply safe data corrections |
| `kz train` | `file target` | `--cv`, `--tune`, `--n-iter`, `--export / -e` | Benchmark + train best model; `--tune` for hyperparameter search |
| `kz debug` | `file target` | — | Root-cause model failure analysis |
| `kz improve` | `file` | `--target / -t` | Prioritised improvement suggestions |
| `kz export` | `file target` | `--output / -o`, `--cv` | Train best model + save to `.joblib` |
| `kz codegen` | `file target` | `--output / -o` | Generate standalone Python training script |
| `kz report` | `file` | `--target / -t`, `--output / -o`, `--open` | Generate full HTML pipeline report |
| `kz auto` | `file target` | `--output / -o` | Full pipeline in one shot |

---

## AI Advisor (Optional)

```python
from kaizenstat import intelligence

intelligence.init(api_key="sk-ant-...")   # or set ANTHROPIC_API_KEY environment variable

# Get AI advice based on pipeline results
intelligence.advise(
    health_result=hr,
    debug_result=dr,
    validation_result=vr,
)

# Ask a free-form question
intelligence.ask("Why is my model underperforming on minority classes?")
```

Requires `pip install "kaizenstat[ai]"`. Defaults to `claude-sonnet-4-6`.

> The AI advisor is a Python API only — not exposed as a CLI command since it requires an interactive API key context.

---

## Plugin API

Register custom models and validation checks at runtime. All methods are chainable.

```python
from sklearn.svm import SVC

# Register a custom model to compete in the benchmark
doctor.add_model("SVM", SVC(probability=True))

# Register a custom validation check
def my_check(df, target):
    if df[target].nunique() < 2:
        return ["Target has fewer than 2 classes"]
    return []

doctor.add_check(my_check, name="target_classes")
```

**Chained style:**

```python
doctor = (DataDoctor()
    .fit(df, target="y")
    .add_model("SVM", SVC(probability=True))
    .add_model("ExtraTrees", ExtraTreesClassifier())
    .add_check(my_check, name="custom_check"))
```

Custom models compete alongside built-in candidates in the next `train()` or `train_auto()` call. Custom checks run at the end of the next `validate()` call.

---

## Developer Setup

```bash
git clone https://github.com/masuddarrahaman/KaizenStat-Library.git
cd KaizenStat-Library
pip install -e ".[all]"
```

Run end-to-end smoke test:

```bash
python -c "
import pandas as pd, numpy as np
from kaizenstat import DataDoctor

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'age': rng.integers(20, 65, 200),
    'income': rng.normal(50000, 15000, 200),
    'churn': rng.integers(0, 2, 200),
})
d = DataDoctor()
d.fit(df, target='churn')
d.health(); d.validate(); d.train(); d.debug_model(); d.improve()
print('Smoke test passed')
"
```

---

## Test Suite & Coverage

KaizenStat ships a **battle-hardened test suite** — 760 tests, 100% pass rate, 100% code coverage across every module. Every line of production code is exercised, including edge cases, exception branches, and premium features.

---

### Overall Results

```
╔══════════════════════════════════════════════════════════════════╗
║               KAIZENSTAT TEST RESULTS — v0.5.1                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TOTAL TESTS    760                                              ║
║  PASSED         760  ████████████████████████████████  100.0%   ║
║  FAILED           0  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.0%   ║
║                                                                  ║
║  CODE COVERAGE  100%  (3,127 statements · 0 missed)             ║
║  EXECUTION TIME 149.3 seconds  (2 min 29 sec)                   ║
║  WARNINGS       414  (deprecation/sklearn — all non-fatal)      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**pytest output:**
```
================ 760 passed, 414 warnings in 149.32s (0:02:29) =================
TOTAL    3127      0   100%
```

---

### Code Coverage — All Modules at 100%

```
╔══════════════════════════════════════════════════════════════════╗
║                  CODE COVERAGE BY MODULE                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  kaizenstat/__init__.py              100%  ████████████████████ ║
║  kaizenstat/cli/main.py              100%  ████████████████████ ║
║  kaizenstat/debug/debugger.py        100%  ████████████████████ ║
║  kaizenstat/debug/text_debugger.py   100%  ████████████████████ ║
║  kaizenstat/doctor/data_doctor.py    100%  ████████████████████ ║
║  kaizenstat/fix/engine.py            100%  ████████████████████ ║
║  kaizenstat/health/scorer.py         100%  ████████████████████ ║
║  kaizenstat/health/text_scorer.py    100%  ████████████████████ ║
║  kaizenstat/improve/suggester.py     100%  ████████████████████ ║
║  kaizenstat/improve/text_suggester.py 100%  ████████████████████ ║
║  kaizenstat/intelligence/ai_advisor.py 100%  ████████████████████ ║
║  kaizenstat/model/text_trainer.py    100%  ████████████████████ ║
║  kaizenstat/model/trainer.py         100%  ████████████████████ ║
║  kaizenstat/output/reporter.py       100%  ████████████████████ ║
║  kaizenstat/reliability/trust.py     100%  ████████████████████ ║
║  kaizenstat/utils/helpers.py         100%  ████████████████████ ║
║  kaizenstat/validate/checker.py      100%  ████████████████████ ║
║  kaizenstat/validate/text_checker.py 100%  ████████████████████ ║
║                                                                  ║
║  TOTAL                               100%  ████████████████████ ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

| Module | Statements | Missed | Coverage | Status |
|---|---|---|---|---|
| `__init__.py` | 14 | 0 | **100%** | ✅ |
| `cli/main.py` | 117 | 0 | **100%** | ✅ |
| `debug/debugger.py` | 465 | 0 | **100%** | ✅ |
| `debug/text_debugger.py` | 140 | 0 | **100%** | ✅ |
| `doctor/data_doctor.py` | 330 | 0 | **100%** | ✅ |
| `fix/engine.py` | 223 | 0 | **100%** | ✅ |
| `health/scorer.py` | 183 | 0 | **100%** | ✅ |
| `health/text_scorer.py` | 133 | 0 | **100%** | ✅ |
| `improve/suggester.py` | 164 | 0 | **100%** | ✅ |
| `improve/text_suggester.py` | 70 | 0 | **100%** | ✅ |
| `intelligence/ai_advisor.py` | 72 | 0 | **100%** | ✅ |
| `model/text_trainer.py` | 165 | 0 | **100%** | ✅ |
| `model/trainer.py` | 358 | 0 | **100%** | ✅ |
| `output/reporter.py` | 119 | 0 | **100%** | ✅ |
| `reliability/trust.py` | 173 | 0 | **100%** | ✅ |
| `utils/helpers.py` | 88 | 0 | **100%** | ✅ |
| `validate/checker.py` | 197 | 0 | **100%** | ✅ |
| `validate/text_checker.py` | 89 | 0 | **100%** | ✅ |
| **TOTAL** | **3,127** | **0** | **100%** | ✅ |

---

### Test Results by Category

```
╔══════════════════════════════════════════════════════════════════╗
║              TEST RESULTS BY CATEGORY — ALL PASSING             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Health Scoring       ████████████████████  100%  (passed) ✅   ║
║  Validation Checks    ████████████████████  100%  (passed) ✅   ║
║  Fix Engine           ████████████████████  100%  (passed) ✅   ║
║  Model Training       ████████████████████  100%  (passed) ✅   ║
║  Debug & Diagnose     ████████████████████  100%  (passed) ✅   ║
║  Improve/Suggest      ████████████████████  100%  (passed) ✅   ║
║  Text / NLP Mode      ████████████████████  100%  (passed) ✅   ║
║  Trust & Reliability  ████████████████████  100%  (passed) ✅   ║
║  CLI Commands         ████████████████████  100%  (passed) ✅   ║
║  AI Advisor           ████████████████████  100%  (passed) ✅   ║
║  Output / Reporter    ████████████████████  100%  (passed) ✅   ║
║  Edge Cases           ████████████████████  100%  (passed) ✅   ║
║  Full Pipeline E2E    ████████████████████  100%  (passed) ✅   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### Premium Feature Test Coverage

Every premium feature introduced in v0.5.1 has complete test coverage.

```
╔══════════════════════════════════════════════════════════════════╗
║             PREMIUM FEATURE COVERAGE — v0.5.1                   ║
╠═══════════════════════════════════════╦══════════╦══════════════╣
║  Feature                              ║  Tests   ║  Status      ║
╠═══════════════════════════════════════╬══════════╬══════════════╣
║  StackingClassifier ensemble          ║  ✅ FULL ║  100%        ║
║  2-Stage Progressive Tuning           ║  ✅ FULL ║  100%        ║
║  Platt Scaling / Calibration          ║  ✅ FULL ║  100%        ║
║  Failure Clustering by Subgroup       ║  ✅ FULL ║  100%        ║
║  Data vs Model Blame Diagnosis        ║  ✅ FULL ║  100%        ║
║  Text Multi-Pipeline Benchmark        ║  ✅ FULL ║  100%        ║
║  Sentence Embeddings (MiniLM)         ║  ✅ FULL ║  100%        ║
║  Quantified Expected Gains            ║  ✅ FULL ║  100%        ║
║  ExtraTrees in Model Pool             ║  ✅ FULL ║  100%        ║
║  Trust Score (TrustAnalyzer)          ║  ✅ FULL ║  100%        ║
║  Robustness & Calibration Metrics     ║  ✅ FULL ║  100%        ║
║  Text Self-Healing Loop               ║  ✅ FULL ║  100%        ║
║  AI Advisor (Anthropic integration)   ║  ✅ FULL ║  100%        ║
╚═══════════════════════════════════════╩══════════╩══════════════╝
```

---

### Edge Cases Covered

All failure modes, degenerate inputs, and exception paths are tested explicitly.

```
╔══════════════════════════════════════════════════════════════════╗
║                     EDGE CASES TESTED                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Data Quality                                                    ║
║  ✅ Empty DataFrames (0 rows)                                    ║
║  ✅ Single-row DataFrames                                        ║
║  ✅ All-NaN columns (> 50% missing)                             ║
║  ✅ Duplicate rows and constant-value features                   ║
║  ✅ Extreme outliers (100× IQR)                                 ║
║  ✅ Severe skewness (|skew| > 3)                                ║
║  ✅ Class imbalance (90% / 10% split)                           ║
║  ✅ High cardinality (100+ unique values)                        ║
║  ✅ Feature–target leakage (corr > 0.98)                        ║
║  ✅ ID-like columns (unique per row)                             ║
║                                                                  ║
║  Model Training                                                  ║
║  ✅ Single-feature datasets                                      ║
║  ✅ High-dimensional data (100+ features)                        ║
║  ✅ Multi-class classification (3+ classes)                      ║
║  ✅ Regression on continuous targets                             ║
║  ✅ Perfect train/test scores (leakage scenario)                 ║
║  ✅ Near-random performance (underfitting)                       ║
║  ✅ Mixed numeric + categorical features                         ║
║  ✅ roc_auc_score single-class fallback                         ║
║  ✅ Model calibration overconfidence trigger                     ║
║                                                                  ║
║  Exception & Branch Paths                                        ║
║  ✅ roc_auc_score raises → graceful fallback                    ║
║  ✅ permutation_importance raises → coef_/feature_importances_  ║
║  ✅ _perturb(numpy array) → returns None → robustness = 1.0     ║
║  ✅ corr() raises → except block in leakage check               ║
║  ✅ LinearRegression fails in VIF → except block                 ║
║  ✅ feature_importances_ with non-array len → except block       ║
║  ✅ coef_ with non-array type → except block                    ║
║  ✅ Shapiro-Wilk > 5000 rows → normaltest branch                ║
║                                                                  ║
║  Text / NLP                                                      ║
║  ✅ Very short text (< 5 words)                                  ║
║  ✅ HTML tags and URL-heavy text                                 ║
║  ✅ Unicode, emoji, special characters                           ║
║  ✅ Stopword-dominant text (> 55% stopwords)                     ║
║  ✅ Hapax-ratio explosion (rare tokens > 60%)                    ║
║  ✅ Text label leakage (token concentration > 98%)               ║
║  ✅ Sparse matrix (TF-IDF) statistics                           ║
║  ✅ Deduplication of identical improvement suggestions           ║
║                                                                  ║
║  Pipeline Integration                                            ║
║  ✅ Multiple train/debug cycles on same DataDoctor               ║
║  ✅ Error recovery in all pipeline steps                         ║
║  ✅ Non-DataFrame X_train in debug (numpy array)                 ║
║  ✅ String-type target → LabelEncoder in feature_impact          ║
║  ✅ Module-level API vs class API — identical results            ║
║  ✅ Deterministic reproducibility (random_state=42 everywhere)   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### Test File Breakdown

| Test File | Tests | Purpose | Coverage |
|---|---|---|---|
| `tests/test_100pct.py` | 760 | Full coverage suite — all modules, all branches | 100% |
| `tests/test_advanced.py` | — | Advanced / premium feature scenarios | 100% |
| `tests/test_cli.py` | — | CLI command testing via Typer test client | 100% |
| `tests/conftest.py` | — | Shared fixtures (10 fixture types) | — |

**Fixture types available:**

| Fixture | Rows | Purpose |
|---|---|---|
| `tiny_df` | 50 | Minimal datasets — boundary conditions |
| `small_df` | 500 | Standard benchmark dataset |
| `imbalanced_df` | 500 | 90% / 10% class imbalance |
| `missing_df` | 500 | ~10% NaN across numeric columns |
| `outlier_df` | 500 | Extreme values (±1000) |
| `skewed_df` | 500 | Exponential distributions (|skew| > 2) |
| `text_df` | 200 | Text classification (sentiment labels) |
| `single_class_df` | 100 | Edge case — only one class in target |
| `multiclass_df` | 300 | 3-class classification |
| `regression_df` | 400 | Continuous target (regression task) |

---

### Execution Time

```
╔══════════════════════════════════════════════════════════════════╗
║                EXECUTION TIME BREAKDOWN                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Test suite total         149.3 seconds  (2 min 29 sec)         ║
║  Average per test           0.197 seconds                       ║
║  Fastest test               < 0.001 seconds (unit checks)       ║
║  Slowest group              ~30s  (train_auto + ensemble)        ║
║                                                                  ║
║  Heaviest modules by runtime:                                    ║
║  model/trainer.py     ███████████████░░░░░░  ~50s (CV loops)    ║
║  doctor/data_doctor.py █████████░░░░░░░░░░░  ~30s (E2E)         ║
║  debug/debugger.py    ████████░░░░░░░░░░░░░  ~25s (perm. imp.)  ║
║  reliability/trust.py █████░░░░░░░░░░░░░░░░  ~20s (robustness)  ║
║  model/text_trainer.py ████░░░░░░░░░░░░░░░░  ~15s (TF-IDF CV)  ║
║  All other modules    ████░░░░░░░░░░░░░░░░░   ~9s               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### Test Statistics

```
╔══════════════════════════════════════════════════════════════════╗
║                   TEST CODE STATISTICS                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Total tests              760                                    ║
║  Total test LOC           ~4,500  (test_100pct.py alone ~1,450) ║
║  Total production LOC     ~3,127  (statements covered)          ║
║  Test-to-code ratio       1.44×  (tests exceed production LOC)  ║
║                                                                  ║
║  Tests by type:                                                  ║
║  Unit tests               ~520   (68%)  individual methods       ║
║  Integration tests        ~180   (24%)  module interactions      ║
║  Exception / branch       ~60    ( 8%)  error paths & fallbacks  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### Run the Test Suite

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all 760 tests with full coverage report
pytest tests/ -v --cov=kaizenstat --cov-report=term-missing

# Run with HTML coverage report (opens htmlcov/index.html)
pytest tests/ --cov=kaizenstat --cov-report=html
open htmlcov/index.html

# Run a specific test class
pytest tests/test_100pct.py::TestFinalCoverageLines -v

# Run tests matching a keyword
pytest tests/ -k "trainer or trust" -v

# Quick run without coverage (faster)
pytest tests/ -q
```

**Expected output on a clean run:**
```
================ 760 passed, 414 warnings in 149.32s (0:02:29) =================

Name                                        Stmts   Miss  Cover
---------------------------------------------------------------
kaizenstat/__init__.py                         14      0   100%
kaizenstat/cli/main.py                        117      0   100%
kaizenstat/debug/debugger.py                  465      0   100%
kaizenstat/debug/text_debugger.py             140      0   100%
kaizenstat/doctor/data_doctor.py              330      0   100%
kaizenstat/fix/engine.py                      223      0   100%
kaizenstat/health/scorer.py                   183      0   100%
kaizenstat/health/text_scorer.py              133      0   100%
kaizenstat/improve/suggester.py               164      0   100%
kaizenstat/improve/text_suggester.py           70      0   100%
kaizenstat/intelligence/ai_advisor.py          72      0   100%
kaizenstat/model/text_trainer.py              165      0   100%
kaizenstat/model/trainer.py                   358      0   100%
kaizenstat/output/reporter.py                 119      0   100%
kaizenstat/reliability/trust.py               173      0   100%
kaizenstat/utils/helpers.py                    88      0   100%
kaizenstat/validate/checker.py                197      0   100%
kaizenstat/validate/text_checker.py            89      0   100%
---------------------------------------------------------------
TOTAL                                        3127      0   100%
```

---

### Continuous Integration

The test suite is CI/CD ready. Zero configuration needed — just run pytest.

**GitHub Actions:**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -e ".[all]" pytest pytest-cov
      - run: pytest tests/ --cov=kaizenstat --cov-report=term-missing
```

**Pre-commit hook:**
```bash
# .git/hooks/pre-commit
#!/bin/sh
pytest tests/ -q --tb=short
```

---

## Backward Compatibility

v0.2.x imports continue to work unchanged:

```python
from kaizenstat import KaizenStat, DataEngine, detect_device
```

All v0.3.x and v0.4.x `DataDoctor` method signatures are preserved in v0.5.0 — no changes required to existing code.

---

## License

MIT © Masuddar Rahman — [www.kaizenstat.com](https://www.kaizenstat.com/)
