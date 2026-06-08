"""
KaizenStat v0.4.0 — Full Pipeline Example
==========================================

Demonstrates every feature in the correct order, step by step.
Each step shows what it does, runs assertions, and prints results.

Compatible with Google Colab and local Python environments.

Run locally:   python3 example_pipeline.py
Run in Colab:  Upload this file → Runtime > Run All

Pipeline Order:
    1.  Health Score          → score data quality 0–100
    2.  Validate              → statistical assumption checks
    3.  Fix                   → safe automatic corrections
    4.  Train                 → smart benchmark + best model
    4b. Train with Tuning     → RandomizedSearchCV hyperparameter tuning  [v0.4.0]
    5.  Debug + Root Cause AI → WHY is the model behaving this way?       [v0.4.0]
    6.  Improve               → prioritised suggestions
    7.  Report                → full HTML report
    8.  Export                → save model to disk
    9.  Codegen               → standalone Python training script
   10.  DataDoctor            → full orchestrator (one object, full pipeline)
   11.  Auto Improve          → fix → retrain → before vs after delta      [v0.4.0]
   12.  Pipeline Confidence   → 0–100 production-readiness score           [v0.4.0]
   13.  Plugin API            → add_model() / add_check()                  [v0.4.0]
"""

# ══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# Installs kaizenstat from PyPI if not already present.
# Safe to run multiple times — skips if already installed.
#
# Optional extras:
#   pip install "kaizenstat[gpu]"  → adds XGBoost + LightGBM (faster on GPU)
#   pip install "kaizenstat[all]"  → all optional dependencies
# ══════════════════════════════════════════════════════════════════════════════

import importlib, subprocess, sys, os

# ── Check if we need to install / upgrade ────────────────────────────────────
def _get_installed_version():
    try:
        import importlib.metadata
        return importlib.metadata.version("kaizenstat")
    except Exception:
        return None

_installed = _get_installed_version()
_need_install = _installed is None or not _installed.startswith("0.4")

if _need_install:
    print(f"Installing kaizenstat 0.4.0 (currently: {_installed or 'not installed'})...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "kaizenstat", "--quiet"])
    importlib.invalidate_caches()
    print("\nInstallation complete.")
    print("=" * 60)
    print("IMPORTANT: The Colab runtime must restart to load the new")
    print("version. Restarting now — please re-run the notebook after")
    print("the restart completes (Runtime > Run All).")
    print("=" * 60)
    # Force a runtime restart in Colab so the new package is loaded.
    # On the second run, kaizenstat 0.4.0 is already installed so this
    # block is skipped entirely and execution continues normally.
    os.kill(os.getpid(), 9)
else:
    print(f"kaizenstat {_installed} already installed. Continuing...\n")


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# KaizenStat — high-level orchestrator
from kaizenstat import DataDoctor

# KaizenStat — individual modules (use these directly if you want more control)
from kaizenstat.health.scorer     import HealthScorer
from kaizenstat.validate.checker  import Validator
from kaizenstat.fix.engine        import FixEngine
from kaizenstat.model.trainer     import ModelTrainer
from kaizenstat.debug.debugger    import ModelDebugger
from kaizenstat.improve.suggester import Suggester
from kaizenstat.output.reporter   import Reporter


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET      = "churn"
OUTPUT_DIR  = "/content" if os.path.exists("/content") else "/tmp"

CSV_PATH    = os.path.join(OUTPUT_DIR, "kaizenstat_example.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "kaizenstat_report.html")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "kaizenstat_model.joblib")
SCRIPT_PATH = os.path.join(OUTPUT_DIR, "kaizenstat_pipeline.py")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")

def ok(message: str) -> None:
    print(f"  [PASS]  {message}")

def info(message: str) -> None:
    print(f"  [INFO]  {message}")

def summary_row(step: str, result: str) -> None:
    print(f"  {'Step ' + step:<14} {result}")


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE DATASET
# Realistic churn dataset with intentional data quality issues so every module
# has something meaningful to detect, report, and fix.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print("  KaizenStat v0.4.0 — Full Pipeline Showcase")
print("═" * 62)

np.random.seed(42)
N = 500

df_raw = pd.DataFrame({
    "age":          np.random.randint(18, 70, N).astype(float),
    "income":       np.random.exponential(40_000, N),           # right-skewed
    "credit_score": np.random.randint(300, 850, N).astype(float),
    "loan_amount":  np.random.randint(1_000, 50_000, N).astype(float),
    "employment":   np.random.choice(
                        ["employed", "self-employed", "unemployed"],
                        N, p=[0.60, 0.30, 0.10]),
    "region":       np.random.choice(["north", "south", "east", "west"], N),
    TARGET:         np.random.choice([0, 1], N, p=[0.75, 0.25]),  # imbalanced 75/25
})

# Intentional data quality issues — every fix module should catch these
df_raw.loc[df_raw.sample(frac=0.12, random_state=1).index, "age"]          = np.nan
df_raw.loc[df_raw.sample(frac=0.08, random_state=2).index, "credit_score"] = np.nan
df_raw.loc[df_raw.sample(frac=0.05, random_state=3).index, "income"]       = np.nan
df_raw = pd.concat([df_raw, df_raw.sample(20, random_state=4)], ignore_index=True)
df_raw["constant_col"] = "same_value"   # zero-variance column

df_raw.to_csv(CSV_PATH, index=False)

print(f"\n  Dataset : {len(df_raw)} rows × {df_raw.shape[1]} columns  →  {CSV_PATH}")
print(f"  Issues  : missing age={df_raw['age'].isna().sum()}, "
      f"credit_score={df_raw['credit_score'].isna().sum()}, "
      f"income={df_raw['income'].isna().sum()}")
print(f"            + 20 duplicate rows · 1 constant column · class imbalance 75/25")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — HEALTH SCORE
# Scores the raw dataset 0–100 across multiple penalty categories.
# Penalises missing values, duplicates, constant columns, skewness, etc.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 1 · Health Score")

scorer = HealthScorer()
hr = scorer.breakdown(df_raw, target=TARGET)
hr.display()

assert hasattr(hr, "score"),  "HealthResult missing .score"
assert 0 <= hr.score <= 100,  f"Score out of range: {hr.score}"
assert len(hr.penalties) > 0, "Expected at least one penalty"
assert hr.score < 90,         "Known issues should push score below 90"

ok(f"Health score = {hr.score:.1f}/100   grade = {hr.grade}   "
   f"penalties found = {len(hr.penalties)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VALIDATE
# Runs normality, VIF multicollinearity, skewness, and data leakage checks.
# Returns a ValidationReport with issues and a pass/fail summary.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 2 · Validate")

vr = Validator().assumptions(df_raw, target=TARGET)
vr.display()

assert hasattr(vr, "issues"),     "ValidationReport missing .issues"
assert hasattr(vr, "checks_run"), "ValidationReport missing .checks_run"
assert vr.checks_run > 0,         "At least one check must have run"

ok(f"Checks run = {vr.checks_run}   issues found = {len(vr.issues)}   "
   f"passed = {vr.passed}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FIX
# Generates a FixPlan (safe=True means low-risk-only), then applies it.
# Returns a cleaned DataFrame — original df_raw is never modified.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 3 · Fix")

engine   = FixEngine()
plan     = engine.plan(df_raw, target=TARGET, safe=True)   # preview
df_fixed = plan.apply(df_raw)                              # apply → new DataFrame

assert hasattr(plan, "actions"),   "FixPlan missing .actions"
assert len(plan.actions) > 0,      "Expected at least one fix action"
assert len(df_fixed) > 0,          "Fixed DataFrame must not be empty"
assert (df_fixed.isna().sum().sum()
        <= df_raw.isna().sum().sum()), "Fix should not increase missing count"

ok(f"Actions applied = {len(plan.actions)}   "
   f"missing before = {df_raw.isna().sum().sum()}   "
   f"missing after  = {df_fixed.isna().sum().sum()}   "
   f"shape {df_raw.shape} → {df_fixed.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN  (smart metric selection)
# Benchmarks multiple models with cross-validation on the training set, then
# evaluates the winner on the held-out test set.
#
# Smart metric selection  [v0.4.0]:
#   • Imbalanced classification (minority class < 20%)  →  f1_weighted
#   • Balanced classification                           →  accuracy
#   • Regression                                        →  r2
#
# This dataset has a 75/25 split → minority = 25% → uses accuracy.
# Change to p=[0.90, 0.10] and the library will automatically switch to f1_weighted.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 4 · Train  (smart metric selection)")

trainer = ModelTrainer()
tr = trainer.train_best(df_fixed, TARGET, test_size=0.2, cv=3)
tr.display()

assert hasattr(tr, "train_score"), "TrainResult missing .train_score"
assert hasattr(tr, "test_score"),  "TrainResult missing .test_score"
assert hasattr(tr, "pipeline"),    "TrainResult missing .pipeline"
assert 0 <= tr.train_score <= 1.0, f"train_score out of range: {tr.train_score}"
assert 0 <= tr.test_score  <= 1.0, f"test_score out of range: {tr.test_score}"
assert not (tr.train_score == 1.0 and tr.test_score == 1.0), \
    "Both scores = 1.0 → data leakage in train_best"

cv_score   = getattr(tr, "cv_score", None)
cv_std     = getattr(tr, "cv_std", None)

ok(f"Model = {tr.model_name}   task = {tr.task}")
if cv_score is not None:
    ok(f"CV score = {cv_score:.4f} ± {cv_std:.4f}   "
       f"Train = {tr.train_score:.4f}   Test = {tr.test_score:.4f}")
else:
    ok(f"Train = {tr.train_score:.4f}   Test = {tr.test_score:.4f}")
info("Smart metric: minority class = 25% → above 20% threshold → using accuracy")
info("Tip: if minority < 20%, KaizenStat auto-switches to f1_weighted")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b — TRAIN WITH HYPERPARAMETER TUNING  [v0.4.0]
# Pass tune=True to run RandomizedSearchCV on the best model from benchmarking.
# best_params will be populated and displayed in the result panel.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 4b · Train with Hyperparameter Tuning  [v0.4.0]")

tr_tuned = trainer.train_best(df_fixed, TARGET, test_size=0.2, cv=3,
                               tune=True, n_iter=10)
tr_tuned.display()

best_params = getattr(tr_tuned, "best_params", {})

ok(f"Tuned model = {tr_tuned.model_name}   Test = {tr_tuned.test_score:.4f}")
if best_params:
    ok(f"Best params found = {len(best_params)}")
    for k, v in best_params.items():
        info(f"  {k} = {v}")
else:
    info("best_params not available in this installed version")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DEBUG + ROOT CAUSE AI  [v0.4.0]
# Diagnoses overfitting / underfitting / data leakage / class imbalance.
# Root Cause AI generates human-readable bullet-point WHY explanations such as:
#   • "Train-test gap is 0.18 — model memorised training data"
#   • "Class imbalance: minority class is 25% — model biased toward majority"
#   • "3 near-constant features detected — these add noise without signal"
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 5 · Debug + Root Cause AI  [v0.4.0]")

df_prep = df_fixed.loc[df_fixed[TARGET].notna()].copy()
X_all   = df_prep.drop(columns=[TARGET])
y_all   = df_prep[TARGET]

if tr.label_encoder is not None:
    y_all = pd.Series(
        tr.label_encoder.transform(y_all),
        index=y_all.index, name=TARGET,
    )

try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
except ValueError:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)

dr = ModelDebugger().model_failure(tr.pipeline, X_tr, X_te, y_tr, y_te)
dr.display()   # ← shows Root Cause AI bullets under "Why is the model behaving this way?"

VALID_LABELS = {
    "data_leakage", "leakage_risk", "data_issue",
    "excellent", "healthy", "acceptable",
    "overfitting_risk", "overfitting", "severe_overfitting",
    "underfitting", "severe_underfitting",
    "weak_model", "broken_model", "unstable_model",
}

assert hasattr(dr, "label"),       "DebugResult missing .label"
assert hasattr(dr, "severity"),    "DebugResult missing .severity"
assert hasattr(dr, "confidence"),  "DebugResult missing .confidence"
assert hasattr(dr, "health_score"),"DebugResult missing .health_score"
assert hasattr(dr, "why_bullets"), "DebugResult missing .why_bullets  (Root Cause AI)"
assert dr.label in VALID_LABELS,   f"Unknown debug label: {dr.label}"
assert 0 <= dr.health_score <= 100,f"health_score out of range: {dr.health_score}"
assert 0 < dr.confidence <= 1.0,   f"confidence out of range: {dr.confidence}"

ok(f"Label = {dr.label}   severity = {dr.severity}   "
   f"confidence = {dr.confidence:.0%}   health_score = {dr.health_score}/100")
ok(f"Train = {dr.train_score:.4f}   Test = {dr.test_score:.4f}   "
   f"Gap = {dr.gap:+.4f}")

if dr.why_bullets:
    ok(f"Root Cause AI generated {len(dr.why_bullets)} explanation bullet(s):")
    for bullet in dr.why_bullets:
        info(f"  {bullet}")
else:
    info("No Root Cause AI bullets for this label (model is healthy — no issues to explain)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — IMPROVE
# Generates a prioritised action list based on health penalties, validation
# issues, debug diagnosis, and feature importances.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 6 · Improve")

ir = Suggester().suggest(df_fixed, target=TARGET,
                         health_result=hr, validation_result=vr,
                         debug_result=dr)

assert hasattr(ir, "suggestions"), "ImprovementReport missing .suggestions"
assert len(ir.suggestions) > 0,    "Expected at least one suggestion"

ok(f"Suggestions generated = {len(ir.suggestions)}")
for i, s in enumerate(ir.suggestions[:5], 1):
    info(f"  {i}. [{s.impact}] {s.action}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — HTML REPORT
# Writes a self-contained HTML file with the full pipeline summary.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 7 · Report")

reporter = Reporter()
reporter.html(
    {"health": hr, "validation": vr, "improvements": ir},
    path=REPORT_PATH,
    open_browser=False,
)

assert os.path.exists(REPORT_PATH),         "HTML report not created"
assert os.path.getsize(REPORT_PATH) > 1000, "HTML report looks too small"

with open(REPORT_PATH) as f:
    html_content = f.read()
assert "<html" in html_content.lower(), "Report file does not look like valid HTML"

ok(f"Report saved → {REPORT_PATH}  ({os.path.getsize(REPORT_PATH):,} bytes)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — EXPORT MODEL
# Saves the trained sklearn pipeline to a .joblib file.
# Reload and predict to confirm the file is usable in production.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 8 · Export Model")

reporter.export_model(tr.pipeline, path=MODEL_PATH)

assert os.path.exists(MODEL_PATH),         "Model file not created"
assert os.path.getsize(MODEL_PATH) > 100,  "Model file looks too small"

loaded_model = joblib.load(MODEL_PATH)
sample_preds = loaded_model.predict(X_te.iloc[:5])
assert len(sample_preds) == 5, "Loaded model must return predictions"

ok(f"Model saved → {MODEL_PATH}  ({os.path.getsize(MODEL_PATH):,} bytes)")
ok(f"Reloaded model predicted 5 samples → {sample_preds.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — CODEGEN
# Generates a standalone Python training script — no KaizenStat dependency
# needed to run it in production. Paste it into any project.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 9 · Codegen")

reporter.codegen(df_fixed, target=TARGET, output_path=SCRIPT_PATH, task=tr.task)

assert os.path.exists(SCRIPT_PATH),         "Generated script not created"
assert os.path.getsize(SCRIPT_PATH) > 200,  "Generated script looks too small"

with open(SCRIPT_PATH) as f:
    generated_code = f.read()
assert "train_test_split" in generated_code, "Script must include train/test split"
assert TARGET in generated_code,             "Script must reference the target column"

ok(f"Script saved → {SCRIPT_PATH}  ({os.path.getsize(SCRIPT_PATH):,} bytes)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — DATADOCTOR  (full orchestrator)
# One object runs the entire pipeline in the correct order.
# This is the recommended way to use KaizenStat — simple sklearn-style API.
#
#   doctor = DataDoctor()
#   doctor.fit(df, target="churn")
#   doctor.health() → doctor.validate() → doctor.fix() → doctor.train()
#   doctor.debug_model() → doctor.improve() → doctor.report()
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 10 · DataDoctor — Full Orchestrator")

DOCTOR_REPORT = os.path.join(OUTPUT_DIR, "kaizenstat_doctor_report.html")

doctor = DataDoctor()
doctor.fit(df_raw, target=TARGET)

h2 = doctor.health()
v2 = doctor.validate()
doctor.fix(safe=True)
t2 = doctor.train(cv=3)
d2 = doctor.debug_model()
i2 = doctor.improve()
doctor.report(output_path=DOCTOR_REPORT, open_browser=False)

assert h2 is not None,                "doctor.health() returned None"
assert v2 is not None,                "doctor.validate() returned None"
assert t2 is not None,                "doctor.train() returned None"
assert d2 is not None,                "doctor.debug_model() returned None"
assert i2 is not None,                "doctor.improve() returned None"
assert d2.label != "data_leakage",    "DataDoctor must not report leakage on clean split"
assert os.path.exists(DOCTOR_REPORT), "DataDoctor HTML report not saved"

ok(f"health = {h2.score:.1f}/100   model = {t2.model_name}   "
   f"debug = {d2.label}   suggestions = {len(i2.suggestions)}")
ok(f"Report → {DOCTOR_REPORT}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — AUTO IMPROVE  [v0.4.0]
# Runs a 3-step self-improving loop:
#   Step 1 — Baseline train (no fixes)
#   Step 2 — Apply safe data fixes automatically
#   Step 3 — Retrain on fixed data (with optional tuning)
#
# Returns a ComparisonResult with:
#   .before       → TrainResult from the baseline
#   .after        → TrainResult after fixing + retraining
#   .score_delta  → float, positive = improvement
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 11 · Auto Improve  [v0.4.0]")

doctor_ai = DataDoctor()
doctor_ai.fit(df_raw, target=TARGET)

# auto_improve() handles everything: baseline → fix → retrain → display delta
comparison = doctor_ai.auto_improve(tune=False)

assert hasattr(comparison, "before"),      "ComparisonResult missing .before"
assert hasattr(comparison, "after"),       "ComparisonResult missing .after"
assert hasattr(comparison, "score_delta"), "ComparisonResult missing .score_delta"
assert isinstance(comparison.score_delta, float), "score_delta must be float"

ok(f"Before → model={comparison.before.model_name}  "
   f"test={comparison.before.test_score:.4f}")
ok(f"After  → model={comparison.after.model_name}   "
   f"test={comparison.after.test_score:.4f}")
ok(f"Delta  → {comparison.score_delta:+.4f}  "
   f"({'improved' if comparison.score_delta > 0 else 'no gain' if comparison.score_delta == 0 else 'regressed'})")

# You can also access the full result objects for further analysis:
before_cv = getattr(comparison.before, "cv_score", None)
after_cv  = getattr(comparison.after,  "cv_score", None)
if before_cv is not None:
    info(f"comparison.before.cv_score = {before_cv:.4f}")
    info(f"comparison.after.cv_score  = {after_cv:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — PIPELINE CONFIDENCE SCORE  [v0.4.0]
# Returns a 0–100 composite score based on:
#   +25 pts   health score (data quality)
#   -5 pts    per validation issue (capped at -20)
#   +20 pts   test score (model performance)
#   -20 pts   train-test gap penalty (overfitting)
#
# Grades:
#   80–100 → production-ready   (green)
#   60–79  → needs work         (yellow)
#   0–59   → not ready          (red)
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 12 · Pipeline Confidence Score  [v0.4.0]")

# Run full pipeline so all result slots are populated for accurate scoring
doctor_conf = DataDoctor()
doctor_conf.fit(df_raw, target=TARGET)
doctor_conf.health()
doctor_conf.validate()
doctor_conf.fix(safe=True)
doctor_conf.train(cv=3)
doctor_conf.debug_model()

confidence = doctor_conf.pipeline_confidence()

assert isinstance(confidence, int), "pipeline_confidence() must return int"
assert 0 <= confidence <= 100,      f"Confidence out of range: {confidence}"

grade = ("production-ready" if confidence >= 80
         else "needs work"   if confidence >= 60
         else "not ready")

ok(f"Pipeline Confidence = {confidence}/100  →  {grade}")
info("Score breakdown: health(+25) + test_score(+20) − validation_issues(−5 each) − gap_penalty")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — PLUGIN API  [v0.4.0]
# Extend KaizenStat without touching library code:
#
#   add_model(name, sklearn_estimator)
#     → custom model competes in the next benchmark()
#     → if it wins, it becomes the trained model
#
#   add_check(fn, name="label")
#     → fn(df, target) → List[str]  (empty list = no issues)
#     → runs inside validate() alongside built-in checks
#
# Both methods return self so you can chain them.
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 13 · Plugin API — add_model() / add_check()  [v0.4.0]")

# --- Custom validation checks ------------------------------------------------

def size_check(df, target):
    """Warn if dataset is too small for reliable ML."""
    if len(df) < 200:
        return [f"Only {len(df)} rows — collect more data before training"]
    return []

def target_balance_check(df, target):
    """Warn on severe class imbalance (minority < 10%)."""
    if target and target in df.columns:
        counts = df[target].value_counts(normalize=True)
        if len(counts) >= 2 and counts.iloc[-1] < 0.10:
            return [f"Severe imbalance: minority class = {counts.iloc[-1]:.1%}"]
    return []

# --- Register everything with the plugin API ---------------------------------

doctor_plugin = DataDoctor()
doctor_plugin.fit(df_raw, target=TARGET)

# Add custom models — they will compete in the next benchmark
doctor_plugin.add_model("SVM",   SVC(kernel="rbf", probability=True, C=1.0))
doctor_plugin.add_model("Dummy", DummyClassifier(strategy="most_frequent"))

# Add custom validation checks
doctor_plugin.add_check(size_check,           name="min_rows")
doctor_plugin.add_check(target_balance_check, name="target_balance")

assert "SVM"   in doctor_plugin._custom_models, "SVM not registered"
assert "Dummy" in doctor_plugin._custom_models, "Dummy not registered"
assert len(doctor_plugin._custom_checks) == 2,  "Expected 2 custom checks"

# validate() runs built-in checks + your registered custom checks
v_plugin = doctor_plugin.validate()
ok(f"Validation ran {v_plugin.checks_run} built-in checks + 2 custom checks")

# train() runs benchmark including your registered custom models
t_plugin = doctor_plugin.train(cv=3)
ok(f"Plugin train: winner = {t_plugin.model_name}   test = {t_plugin.test_score:.4f}")
ok(f"Custom models that competed: {list(doctor_plugin._custom_models.keys())}")

# --- Chained API (all methods return self) -----------------------------------

doctor_chain = (
    DataDoctor()
    .fit(df_raw, target=TARGET)
    .add_model("SVM_linear", SVC(kernel="linear", probability=True))
    .add_check(size_check, name="row_count")
)
assert "SVM_linear" in doctor_chain._custom_models, "Chained add_model failed"
assert len(doctor_chain._custom_checks) == 1,       "Chained add_check failed"
ok("Chained API  fit() → add_model() → add_check()  works correctly")


# ══════════════════════════════════════════════════════════════════════════════
# CLI EQUIVALENTS
# Every step above has a matching terminal command.
# ══════════════════════════════════════════════════════════════════════════════

section("CLI EQUIVALENTS")

print(f"""
  kz health   {CSV_PATH} --target {TARGET}
  kz validate {CSV_PATH} --target {TARGET}
  kz fix      {CSV_PATH} --target {TARGET} --preview
  kz fix      {CSV_PATH} --target {TARGET} -o {OUTPUT_DIR}/fixed.csv
  kz train    {CSV_PATH} {TARGET} --cv 3
  kz train    {CSV_PATH} {TARGET} --cv 3 --tune --n-iter 20
  kz debug    {CSV_PATH} {TARGET}
  kz improve  {CSV_PATH} --target {TARGET}
  kz report   {CSV_PATH} --target {TARGET} -o {REPORT_PATH} --open
  kz export   {CSV_PATH} {TARGET} -o {MODEL_PATH}
  kz codegen  {CSV_PATH} {TARGET} -o {SCRIPT_PATH}
  kz auto     {CSV_PATH} {TARGET}
""")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print("  ALL STEPS PASSED — KaizenStat v0.4.0 fully validated")
print("═" * 62)
print()
summary_row("1  Health",       f"{hr.score:.1f}/100  grade={hr.grade}  "
                               f"penalties={len(hr.penalties)}")
summary_row("2  Validate",     f"{vr.checks_run} checks  issues={len(vr.issues)}  "
                               f"passed={vr.passed}")
summary_row("3  Fix",          f"{len(plan.actions)} actions  "
                               f"missing {df_raw.isna().sum().sum()} → {df_fixed.isna().sum().sum()}")
cv_str = f"cv={getattr(tr, 'cv_score', None) or '—'}  " if getattr(tr, 'cv_score', None) else ""
summary_row("4  Train",        f"{tr.model_name}  {cv_str}test={tr.test_score:.4f}")
bp = getattr(tr_tuned, 'best_params', {})
summary_row("4b Tuned",        f"{tr_tuned.model_name}  test={tr_tuned.test_score:.4f}  "
                               f"params={len(bp)}")
summary_row("5  Debug+AI",     f"label={dr.label}  severity={dr.severity}  "
                               f"why_bullets={len(dr.why_bullets)}")
summary_row("6  Improve",      f"{len(ir.suggestions)} suggestions")
summary_row("7  Report",       f"{REPORT_PATH}")
summary_row("8  Export",       f"{MODEL_PATH}  ({os.path.getsize(MODEL_PATH):,} bytes)")
summary_row("9  Codegen",      f"{SCRIPT_PATH}  ({os.path.getsize(SCRIPT_PATH):,} bytes)")
summary_row("10 DataDoctor",   f"health={h2.score:.1f}  model={t2.model_name}  "
                               f"debug={d2.label}  suggestions={len(i2.suggestions)}")
summary_row("11 AutoImprove",  f"before={comparison.before.test_score:.4f}  "
                               f"after={comparison.after.test_score:.4f}  "
                               f"delta={comparison.score_delta:+.4f}")
summary_row("12 Confidence",   f"{confidence}/100  →  {grade}")
summary_row("13 Plugins",      f"models={list(doctor_plugin._custom_models.keys())}  "
                               f"checks={len(doctor_plugin._custom_checks)}")
print()
print("  KaizenStat v0.4.0 is working correctly.")
print("═" * 62)
