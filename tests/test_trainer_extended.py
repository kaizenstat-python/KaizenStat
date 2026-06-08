"""Extended ModelTrainer tests — evaluate, fallback split, module APIs, edge paths."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from kaizenstat.model import trainer as trainer_mod
from kaizenstat.model.trainer import ModelTrainer


# ── evaluate() ────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_evaluate_classification(self, small_df):
        from sklearn.model_selection import train_test_split
        X = small_df.drop(columns=["churn"])
        y = small_df["churn"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = ModelTrainer().train_best(small_df, target="churn", cv=3)
        metrics = ModelTrainer().evaluate(result.pipeline, X_test, y_test)
        assert "accuracy" in metrics

    def test_evaluate_regression(self, regression_df):
        from sklearn.model_selection import train_test_split
        X = regression_df.drop(columns=["target"])
        y = regression_df["target"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = ModelTrainer().train_best(regression_df, target="target", cv=3)
        metrics = ModelTrainer().evaluate(result.pipeline, X_test, y_test, task="regression")
        assert "r2" in metrics

    def test_evaluate_auto_detects_task(self, small_df):
        result = ModelTrainer().train_best(small_df, target="churn", cv=3)
        from sklearn.model_selection import train_test_split
        X = small_df.drop(columns=["churn"])
        y = small_df["churn"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # task=None triggers auto-detection
        metrics = ModelTrainer().evaluate(result.pipeline, X_test, y_test, task=None)
        assert isinstance(metrics, dict)


# ── Module-level API ──────────────────────────────────────────────────────────

class TestTrainerModuleAPI:
    def test_module_benchmark(self, small_df):
        result = trainer_mod.benchmark(small_df, target="churn", cv=3)
        assert result.best_name is not None

    def test_module_train_best(self, small_df):
        result = trainer_mod.train_best(small_df, target="churn", cv=3)
        assert result.test_score > 0

    def test_module_train_auto(self, small_df):
        result = trainer_mod.train_auto(small_df, target="churn", cv=3, ensemble=False)
        assert result is not None

    def test_module_evaluate(self, small_df):
        result = trainer_mod.train_best(small_df, target="churn", cv=3)
        from sklearn.model_selection import train_test_split
        X = small_df.drop(columns=["churn"])
        y = small_df["churn"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics = trainer_mod.evaluate(result.pipeline, X_test, y_test)
        assert "accuracy" in metrics


# ── extra_models in benchmark ─────────────────────────────────────────────────

class TestExtraModels:
    def test_benchmark_with_extra_model(self, small_df):
        from sklearn.tree import DecisionTreeClassifier
        extra = {"DT": DecisionTreeClassifier(max_depth=3, random_state=0)}
        result = ModelTrainer().benchmark(small_df, target="churn", cv=3, extra_models=extra)
        names = [e.name for e in result.entries]
        assert "DT" in names

    def test_train_best_with_extra_model(self, small_df):
        from sklearn.tree import DecisionTreeClassifier
        extra = {"DT": DecisionTreeClassifier(max_depth=3, random_state=0)}
        result = ModelTrainer().train_best(small_df, target="churn", cv=3, extra_models=extra)
        assert result is not None


# ── Fallback train split (unstratifiable) ─────────────────────────────────────

class TestFallbackSplit:
    def test_fallback_on_rare_class(self):
        """A class with only 1 sample → stratify fails → fallback to non-stratified."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 50),
            "target": [0] * 49 + [1],  # 1 positive sample only
        })
        result = ModelTrainer().train_best(df, target="target", cv=2)
        assert result is not None


# ── _select_metric ────────────────────────────────────────────────────────────

class TestSelectMetric:
    def test_balanced_uses_accuracy(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        assert ModelTrainer._select_metric("classification", y) == "accuracy"

    def test_imbalanced_uses_f1(self):
        y = pd.Series([0] * 19 + [1])  # minority = 5%
        assert ModelTrainer._select_metric("classification", y) == "f1_weighted"

    def test_regression_uses_r2(self):
        y = pd.Series([1.0, 2.0, 3.0])
        assert ModelTrainer._select_metric("regression", y) == "r2"


# ── _make_ohe ─────────────────────────────────────────────────────────────────

class TestMakeOHE:
    def test_make_ohe_returns_encoder(self):
        ohe = ModelTrainer._make_ohe()
        assert hasattr(ohe, "fit_transform")


# ── _analyze_data ─────────────────────────────────────────────────────────────

class TestAnalyzeData:
    def test_analyze_returns_profile(self):
        X = pd.DataFrame({"a": range(100), "b": range(100)})
        y = pd.Series([i % 2 for i in range(100)])
        profile = ModelTrainer._analyze_data(X, y)
        assert "n_rows" in profile
        assert "imbalance" in profile
        assert "high_dim" in profile

    def test_analyze_high_dim(self):
        X = pd.DataFrame(np.random.randn(100, 60), columns=[f"f{i}" for i in range(60)])
        y = pd.Series([i % 2 for i in range(100)])
        profile = ModelTrainer._analyze_data(X, y)
        assert profile["high_dim"] is True


# ── TrainResult display ────────────────────────────────────────────────────────

class TestTrainResultDisplay:
    def test_display_with_tune_params(self, small_df):
        result = ModelTrainer().train_best(small_df, target="churn", cv=3, tune=True, n_iter=5)
        result.display()  # should not raise

    def test_display_regression(self, regression_df):
        result = ModelTrainer().train_best(regression_df, target="target", cv=3)
        result.display()


# ── BenchmarkResult display ───────────────────────────────────────────────────

class TestBenchmarkResultDisplay:
    def test_benchmark_display(self, small_df):
        result = ModelTrainer().benchmark(small_df, target="churn", cv=3)
        result.display()

    def test_benchmark_with_error_entry(self):
        """Entry with error=str should still render in table."""
        from kaizenstat.model.trainer import BenchmarkResult, ModelEntry
        entry = ModelEntry(name="Bad", score=0.0, std=0.0, train_time=0.1,
                           metric="accuracy", error="Failed: some error")
        result = BenchmarkResult(task="classification", metric="accuracy",
                                  entries=[entry], best_name="Bad", best_score=0.0,
                                  best_pipeline=None, label_encoder=None)
        result.display()


# ── Multiclass ROC AUC ────────────────────────────────────────────────────────

class TestMulticlassMetrics:
    def test_multiclass_metrics_computed(self, multiclass_df):
        result = ModelTrainer().train_best(multiclass_df, target="target", cv=3)
        assert result.metrics is not None
        assert "accuracy" in result.metrics
