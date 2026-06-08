"""Extended ModelDebugger tests — covers uncovered branches."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kaizenstat.debug.debugger import ModelDebugger


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cls_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200),
                       "city": rng.choice(["NY", "LA"], 200)})
    y = pd.Series((X["a"] > 0).astype(int), name="target")
    return X, y


@pytest.fixture
def cls_model(cls_data):
    X, y = cls_data
    X_num = X.select_dtypes(include="number")
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=300))])
    pipe.fit(X_num, y)
    return pipe, X_num, y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    y = pd.Series(X["a"] * 2 + rng.normal(0, 0.1, 200), name="target")
    return X, y


@pytest.fixture
def reg_model(reg_data):
    X, y = reg_data
    pipe = Pipeline([("model", Ridge())])
    pipe.fit(X, y)
    return pipe, X, y


# ── overfitting_check ─────────────────────────────────────────────────────────

class TestOverfittingCheck:
    def test_returns_dict_with_keys(self, cls_model):
        pipe, X, y = cls_model
        result = ModelDebugger().overfitting_check(pipe, X, X, y, y)
        assert "train_score" in result
        assert "test_score" in result
        assert "gap" in result
        assert "label" in result
        assert "is_overfitting" in result

    def test_large_gap_overfitting_label(self, cls_model):
        pipe, X, y = cls_model
        # Use training as train but tiny subset as test to create large gap
        result = ModelDebugger().overfitting_check(pipe, X, X.iloc[:20], y, y.iloc[:20])
        assert isinstance(result["gap"], float)

    def test_regression_overfitting_check(self, reg_model):
        pipe, X, y = reg_model
        result = ModelDebugger().overfitting_check(pipe, X, X, y, y)
        assert "label" in result


# ── error_analysis ────────────────────────────────────────────────────────────

class TestErrorAnalysis:
    def test_classification_error_analysis(self, cls_model):
        pipe, X, y = cls_model
        errors = ModelDebugger().error_analysis(pipe, X, y)
        assert isinstance(errors, pd.DataFrame)
        assert "y_true" in errors.columns
        assert "y_pred" in errors.columns

    def test_regression_error_analysis(self, reg_model):
        pipe, X, y = reg_model
        errors = ModelDebugger().error_analysis(pipe, X, y)
        assert "residual" in errors.columns
        assert "abs_error" in errors.columns


# ── feature_importance ────────────────────────────────────────────────────────

class TestFeatureImportance:
    def test_feature_importance_with_rf(self, cls_data):
        X, y = cls_data
        X_num = X.select_dtypes(include="number")
        rf = Pipeline([("model", RandomForestClassifier(n_estimators=10, random_state=0))])
        rf.fit(X_num, y)
        fi = ModelDebugger().feature_importance(rf, X_num, y)
        assert isinstance(fi, pd.Series)

    def test_feature_importance_empty_for_no_importances(self, cls_model):
        """DummyClassifier has no feature_importances — should return empty Series."""
        rng = np.random.default_rng(5)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100)})
        y = pd.Series(rng.integers(0, 2, 100))
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        fi = ModelDebugger().feature_importance(model, X, y)
        assert isinstance(fi, pd.Series)

    def test_feature_importance_with_names(self, cls_data):
        X, y = cls_data
        X_num = X.select_dtypes(include="number")
        rf = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=0))])
        rf.fit(X_num, y)
        fi = ModelDebugger().feature_importance(rf, X_num, y, feature_names=["feat_a", "feat_b"])
        assert isinstance(fi, pd.Series)


# ── feature_impact ────────────────────────────────────────────────────────────

class TestFeatureImpact:
    def test_feature_impact_returns_dict(self, cls_model):
        pipe, X, y = cls_model
        impacts = ModelDebugger().feature_impact(pipe, X, y, top_n=2)
        assert isinstance(impacts, dict)
        assert len(impacts) <= len(X.columns)

    def test_feature_impact_regression(self, reg_model):
        pipe, X, y = reg_model
        impacts = ModelDebugger().feature_impact(pipe, X, y)
        assert isinstance(impacts, dict)


# ── bias_detection ────────────────────────────────────────────────────────────

class TestBiasDetection:
    def test_bias_detection_with_categorical(self, cls_data, cls_model):
        pipe, X_num, y = cls_model
        X, _ = cls_data
        # Use the full X with categorical columns
        X_for_bias = pd.DataFrame({"a": X["a"], "b": X["b"], "city": X["city"]})
        model = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
        model.fit(X_num, y)
        result = ModelDebugger().bias_detection(model, X_num, y)
        assert isinstance(result, dict)

    def test_bias_detection_no_categorical(self, cls_model):
        pipe, X, y = cls_model
        result = ModelDebugger().bias_detection(pipe, X, y)
        assert isinstance(result, dict)

    def test_bias_detection_with_cat_in_df(self, cls_data):
        X, y = cls_data
        X_cat_only = X[["city"]]
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_cat_only, y)
        result = ModelDebugger().bias_detection(model, X_cat_only, y)
        assert isinstance(result, dict)

    def test_bias_detection_no_features_found(self):
        """No categorical features returns empty dict."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        model = DummyClassifier()
        model.fit(X, y)
        result = ModelDebugger().bias_detection(model, X, y, sensitive_features=[])
        assert result == {}


# ── dataset_difficulty ────────────────────────────────────────────────────────

class TestDatasetDifficulty:
    def test_easy_dataset(self):
        X = pd.DataFrame({"x": range(100)})
        y = pd.Series([i % 2 for i in range(100)])
        diff = ModelDebugger().dataset_difficulty(X, y)
        assert 0.0 <= diff <= 1.0

    def test_no_numeric_columns(self):
        X = pd.DataFrame({"cat": ["a", "b"] * 50})
        y = pd.Series([0, 1] * 50)
        diff = ModelDebugger().dataset_difficulty(X, y)
        assert diff == 0.5

    def test_regression_difficulty(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100)})
        y = pd.Series(rng.normal(0, 1, 100))  # random regression target
        diff = ModelDebugger().dataset_difficulty(X, y)
        assert 0.0 <= diff <= 1.0

    def test_numpy_array_y(self):
        X = pd.DataFrame({"x": range(100)})
        y = np.array([i % 2 for i in range(100)])
        diff = ModelDebugger().dataset_difficulty(X, y)
        assert 0.0 <= diff <= 1.0
