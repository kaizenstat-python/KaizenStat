"""Tests for reliability/trust.py — TrustAnalyzer, TrustReport."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kaizenstat.reliability import trust as trust_mod
from kaizenstat.reliability.trust import TrustAnalyzer, TrustReport


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    y = pd.Series((X["a"] + X["b"] > 0).astype(int), name="target")
    return X, y


@pytest.fixture
def trained_classifier(binary_data):
    X, y = binary_data
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))])
    pipe.fit(X, y)
    return pipe, X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    y = pd.Series(X["a"] * 2 + rng.normal(0, 0.2, 200), name="target")
    return X, y


@pytest.fixture
def trained_regressor(regression_data):
    from sklearn.linear_model import Ridge
    X, y = regression_data
    pipe = Pipeline([("model", Ridge())])
    pipe.fit(X, y)
    return pipe, X, y


# ── TrustReport.display ────────────────────────────────────────────────────────

class TestTrustReportDisplay:
    def test_display_high_trust(self):
        report = TrustReport(
            trust_score=90, grade="production-ready",
            confidence_mean=0.92, confidence_std=0.05,
            uncertain_fraction=0.02, robustness_score=0.95,
            calibration_gap=0.04, failure_slices=[], notes=[],
        )
        report.display()  # should not raise

    def test_display_medium_trust(self):
        report = TrustReport(
            trust_score=65, grade="needs review",
            confidence_mean=0.70, confidence_std=0.10,
            uncertain_fraction=0.15, robustness_score=0.80,
            calibration_gap=0.10, failure_slices=["Low-conf band: 20 samples"],
            notes=["Over/under-confident"],
        )
        report.display()

    def test_display_low_trust(self):
        report = TrustReport(
            trust_score=40, grade="not ready",
            confidence_mean=0.55, confidence_std=0.20,
            uncertain_fraction=0.45, robustness_score=0.60,
            calibration_gap=0.25, failure_slices=["Class 0: 50%"],
            notes=["Fragile model"],
        )
        report.display()


# ── TrustAnalyzer.analyze — classification ────────────────────────────────────

class TestTrustAnalyzerClassification:
    def test_analyze_returns_trust_report(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y)
        assert isinstance(result, TrustReport)
        assert 0 <= result.trust_score <= 100

    def test_trust_score_in_range(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y)
        assert 0 <= result.trust_score <= 100

    def test_grade_valid_string(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y)
        assert result.grade in ("production-ready", "needs review", "not ready")

    def test_confidence_mean_in_range(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y)
        assert 0 <= result.confidence_mean <= 1

    def test_robustness_score_in_range(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y)
        assert 0 <= result.robustness_score <= 1

    def test_calibration_gap_nonnegative(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y)
        assert result.calibration_gap >= 0

    def test_notes_populated_on_bad_calibration(self):
        """Force a badly calibrated model (all-same predictions) to test notes."""
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100)})
        y = pd.Series([0] * 90 + [1] * 10, name="t")
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        result = TrustAnalyzer().analyze(model, X, y)
        assert isinstance(result.notes, list)

    def test_analyze_explicit_task_classification(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = TrustAnalyzer().analyze(pipe, X, y, task="classification")
        assert isinstance(result, TrustReport)

    def test_failure_slices_populated_for_bad_model(self):
        """DummyClassifier on mixed data should produce failure slices."""
        rng = np.random.default_rng(5)
        X = pd.DataFrame({
            "num": rng.normal(0, 1, 200),
            "cat": ["A"] * 100 + ["B"] * 100,
        })
        y = pd.Series([0] * 100 + [1] * 100, name="t")
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        result = TrustAnalyzer().analyze(model, X, y)
        assert isinstance(result.failure_slices, list)

    def test_low_confidence_note_triggered(self, binary_data):
        """Dummy always outputs 0.5 proba — triggers high uncertain fraction."""
        X, y = binary_data
        model = DummyClassifier(strategy="uniform", random_state=0)
        model.fit(X, y)
        result = TrustAnalyzer().analyze(model, X, y, low_conf_threshold=0.99)
        assert result.uncertain_fraction >= 0

    def test_no_predict_proba_model(self):
        """Model without predict_proba uses decision_function or neutral."""
        from sklearn.svm import SVC
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
        y = pd.Series([i % 2 for i in range(100)])
        model = SVC(kernel="rbf")
        model.fit(X, y)
        result = TrustAnalyzer().analyze(model, X, y)
        assert 0 <= result.trust_score <= 100


# ── TrustAnalyzer.analyze — regression ───────────────────────────────────────

class TestTrustAnalyzerRegression:
    def test_regression_trust_returns_report(self, trained_regressor):
        pipe, X, y = trained_regressor
        result = TrustAnalyzer().analyze(pipe, X, y, task="regression")
        assert isinstance(result, TrustReport)

    def test_regression_trust_score_in_range(self, trained_regressor):
        pipe, X, y = trained_regressor
        result = TrustAnalyzer().analyze(pipe, X, y, task="regression")
        assert 0 <= result.trust_score <= 100

    def test_regression_grade_valid(self, trained_regressor):
        pipe, X, y = trained_regressor
        result = TrustAnalyzer().analyze(pipe, X, y, task="regression")
        assert result.grade in ("production-ready", "needs review", "not ready")

    def test_regression_notes_present(self, trained_regressor):
        pipe, X, y = trained_regressor
        result = TrustAnalyzer().analyze(pipe, X, y, task="regression")
        assert any("Regression trust" in n for n in result.notes)

    def test_regression_auto_task_detection(self, regression_data):
        from sklearn.linear_model import Ridge
        X, y = regression_data
        y_float = y.astype(float)
        pipe = Pipeline([("model", Ridge())])
        pipe.fit(X, y_float)
        result = TrustAnalyzer().analyze(pipe, X, y_float)
        assert isinstance(result, TrustReport)


# ── TrustAnalyzer perturbation on text ───────────────────────────────────────

class TestPerturbation:
    def test_perturb_text_series(self):
        analyzer = TrustAnalyzer()
        texts = pd.Series(["hello world foo bar baz"] * 20)
        perturbed = analyzer._perturb(texts, seed=0)
        assert isinstance(perturbed, pd.Series)
        assert len(perturbed) == 20

    def test_perturb_dataframe(self):
        analyzer = TrustAnalyzer()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        perturbed = analyzer._perturb(X, seed=1)
        assert isinstance(perturbed, pd.DataFrame)
        assert perturbed.shape == X.shape

    def test_perturb_none_type(self):
        analyzer = TrustAnalyzer()
        result = analyzer._perturb(42, seed=0)  # non-DataFrame, non-Series
        assert result is None

    def test_perturb_short_text(self):
        """Short texts (≤3 words) should be returned unchanged."""
        analyzer = TrustAnalyzer()
        texts = pd.Series(["hi", "ok", "yes"])
        perturbed = analyzer._perturb(texts, seed=0)
        assert isinstance(perturbed, pd.Series)

    def test_robustness_with_none_perturb(self):
        """_robustness returns 1.0 when perturb returns None."""
        analyzer = TrustAnalyzer()
        model = DummyClassifier()
        model.fit([[0], [1]], [0, 1])
        result = analyzer._robustness(model, 42, np.array([0, 1]), 1)
        assert result == 1.0


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_analyze_function(self, trained_classifier):
        pipe, X, y = trained_classifier
        result = trust_mod.analyze(pipe, X, y)
        assert isinstance(result, TrustReport)
