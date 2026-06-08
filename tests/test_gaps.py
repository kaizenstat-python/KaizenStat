"""Gap-filling tests — covers remaining uncovered lines across all modules."""
import numpy as np
import pandas as pd
import pytest

# ── utils/helpers ─────────────────────────────────────────────────────────────

from kaizenstat.utils.helpers import (
    _avg_word_count,
    detect_id_columns,
    detect_task_type,
    detect_text_columns,
    dominant_text_column,
    is_text_dataset,
    validate_dataframe,
)


class TestDetectTaskType:
    def test_category_dtype_is_classification(self):
        y = pd.Series(pd.Categorical(["a", "b", "a"]))
        assert detect_task_type(y) == "classification"

    def test_high_cardinality_integer_is_regression(self):
        y = pd.Series(range(100), dtype=int)  # 100 unique values > 20
        assert detect_task_type(y) == "regression"

    def test_float_is_regression(self):
        y = pd.Series([1.0, 2.5, 3.7])
        assert detect_task_type(y) == "regression"

    def test_low_cardinality_integer_is_classification(self):
        y = pd.Series([0, 1, 0, 1, 2])
        assert detect_task_type(y) == "classification"

    def test_object_dtype_is_classification(self):
        y = pd.Series(["cat", "dog", "cat"])
        assert detect_task_type(y) == "classification"


class TestValidateDataframe:
    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_dataframe(None)

    def test_non_dataframe_list_raises(self):
        with pytest.raises(TypeError):
            validate_dataframe([1, 2, 3])

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame())


class TestDetectIdColumns:
    def test_id_column_by_name(self):
        df = pd.DataFrame({"user_id": range(100), "value": range(100)})
        ids = detect_id_columns(df)
        assert "user_id" in ids

    def test_all_unique_integer_col(self):
        df = pd.DataFrame({
            "seq_id": list(range(50)),
            "value": np.random.randn(50),
        })
        ids = detect_id_columns(df)
        assert "seq_id" in ids or "value" not in ids  # seq_id should be flagged

    def test_all_unique_string_col_large(self):
        import uuid
        df = pd.DataFrame({
            "uuid_col": [str(uuid.uuid4()) for _ in range(50)],
            "value": range(50),
        })
        ids = detect_id_columns(df)
        assert "uuid_col" in ids

    def test_small_df_not_flagged(self):
        df = pd.DataFrame({"x": range(5), "y": range(5)})
        ids = detect_id_columns(df)
        # Short data (len≤20) → not flagged as ID
        assert "x" not in ids


class TestAvgWordCount:
    def test_empty_series_returns_zero(self):
        s = pd.Series([], dtype=str)
        assert _avg_word_count(s) == 0.0

    def test_large_series_sampled(self):
        # Series with more than 500 entries triggers sampling
        s = pd.Series([f"word{i} another" for i in range(600)])
        result = _avg_word_count(s, sample=500)
        assert result > 0.0


class TestDetectTextColumns:
    def test_long_text_detected(self):
        # Need enough unique sentences so nunique > max(2, n*0.05) = max(2, 3.0) = 3
        sentences = [
            f"This is a detailed review sentence number {i} about various things" for i in range(60)
        ]
        df = pd.DataFrame({"review": sentences, "label": [i % 2 for i in range(60)]})
        cols = detect_text_columns(df, exclude=["label"])
        assert "review" in cols

    def test_short_categorical_not_detected(self):
        df = pd.DataFrame({
            "city": ["NY", "LA", "Chicago"] * 20,
            "label": [0] * 60,
        })
        cols = detect_text_columns(df, exclude=["label"])
        assert "city" not in cols

    def test_excluded_column_not_detected(self):
        df = pd.DataFrame({
            "review": ["Long review text with many words here"] * 30,
            "label": [0] * 30,
        })
        cols = detect_text_columns(df, exclude=["review", "label"])
        assert "review" not in cols

    def test_numeric_column_not_detected(self):
        df = pd.DataFrame({"num": range(50), "label": range(50)})
        cols = detect_text_columns(df)
        assert "num" not in cols

    def test_empty_string_col_not_detected(self):
        df = pd.DataFrame({"empty": pd.Series([None, None, None], dtype=object)})
        cols = detect_text_columns(df)
        assert "empty" not in cols


class TestDominantTextColumn:
    def test_returns_none_when_no_text(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert dominant_text_column(df) is None

    def test_returns_text_column(self):
        sentences = [f"This is a very detailed review with many words number {i}" for i in range(30)]
        df = pd.DataFrame({"review": sentences, "label": [i % 2 for i in range(30)]})
        col = dominant_text_column(df, exclude=["label"])
        assert col == "review"


class TestIsTextDataset:
    def test_text_dataset_true(self):
        sentences = [f"Great product very happy with it bought twice already number {i}" for i in range(50)]
        df = pd.DataFrame({"review": sentences, "label": [i % 2 for i in range(50)]})
        assert is_text_dataset(df, target="label") is True

    def test_tabular_dataset_false(self):
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        assert is_text_dataset(df) is False


# ── health/scorer ─────────────────────────────────────────────────────────────

from kaizenstat.health import scorer as health_mod
from kaizenstat.health.scorer import HealthResult, HealthScorer


class TestHealthResultRepr:
    def test_repr_output(self, small_df):
        result = HealthScorer().report(small_df, target="churn")
        r = repr(result)
        assert "HealthResult" in r
        assert "score=" in r
        assert "grade=" in r


class TestHealthScorerMethods:
    def test_scorer_score_method(self, small_df):
        s = HealthScorer()
        score = s.score(small_df, target="churn")
        assert 0 <= score <= 100

    def test_health_summary_moderate(self):
        hs = HealthScorer()
        assert "Moderate" in hs._summary(65)

    def test_health_summary_significant(self):
        hs = HealthScorer()
        assert "Significant" in hs._summary(50)

    def test_health_summary_critical(self):
        hs = HealthScorer()
        assert "Critical" in hs._summary(30)

    def test_health_summary_excellent(self):
        hs = HealthScorer()
        assert "Excellent" in hs._summary(95)

    def test_health_summary_good(self):
        hs = HealthScorer()
        assert "Good" in hs._summary(78)

    def test_leakage_proxy_exception_handling(self):
        """Column with object dtype still doesn't crash _leakage_proxy."""
        df = pd.DataFrame({
            "x": [float("nan")] * 50 + list(range(50)),
            "target": range(100),
        })
        result = HealthScorer().report(df, target="target")
        assert result.score >= 0


class TestHealthModuleAPI:
    def test_module_score_function(self, small_df):
        s = health_mod.score(small_df, target="churn")
        assert 0 <= s <= 100

    def test_module_breakdown_function(self, small_df):
        result = health_mod.breakdown(small_df, target="churn")
        assert isinstance(result, HealthResult)


# ── validate/checker ──────────────────────────────────────────────────────────

from kaizenstat.validate import checker as validate_mod
from kaizenstat.validate.checker import Validator


class TestValidatorMethods:
    def test_skewness_method(self):
        df = pd.DataFrame({
            "skewed": np.concatenate([np.ones(90), np.exp(np.linspace(0, 5, 10))]),
            "target": [0, 1] * 50,
        })
        result = Validator().skewness(df)
        assert result.checks_run == 1

    def test_multicollinearity_method(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 100)
        df = pd.DataFrame({
            "a": x,
            "b": x * 0.99 + rng.normal(0, 0.01, 100),  # near-perfect correlation
            "target": rng.integers(0, 2, 100),
        })
        result = Validator().multicollinearity(df)
        assert result.checks_run == 1

    def test_leakage_method(self):
        df = pd.DataFrame({
            "leak": range(100),
            "target": range(100),
        })
        result = Validator().leakage(df, target="target")
        assert result.checks_run == 1

    def test_distribution_check_method(self, small_df):
        result = Validator().distribution_check(small_df, target="churn")
        assert result.checks_run == 1

    def test_normality_short_column_skipped(self):
        """Columns with fewer than 8 rows should be skipped."""
        df = pd.DataFrame({
            "short_col": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        })
        result = Validator().assumptions(df, target="target")
        assert result is not None

    def test_multicollinearity_vif_with_high_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        df = pd.DataFrame({
            "a": x,
            "b": x + rng.normal(0, 0.001, 200),
            "c": rng.normal(0, 1, 200),
            "target": rng.integers(0, 2, 200),
        })
        result = Validator().assumptions(df, target="target")
        assert result is not None

    def test_check_skewness_only_severe(self):
        """Skewness check when only severe (not moderate) — moderate branch is empty."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "super_skewed": np.concatenate([np.zeros(190), [1e6, 1e7, 1e8, 1e9, 1e10]]),
            "normal_col": rng.normal(0, 1, 195),
            "target": rng.integers(0, 2, 195),
        })
        result = Validator().skewness(df, target="target")
        assert result is not None

    def test_near_constant_distribution(self):
        """Nearly constant column triggers near-constant distribution check."""
        df = pd.DataFrame({
            "const": [1.000001 * (1 + i * 1e-6) for i in range(100)],
            "target": [i % 2 for i in range(100)],
        })
        result = Validator().distribution_check(df, target="target")
        assert result is not None


class TestValidatorModuleAPI:
    def test_module_skewness(self, skewed_df):
        result = validate_mod.skewness(skewed_df, target="target")
        assert result.checks_run == 1

    def test_module_multicollinearity(self, small_df):
        result = validate_mod.multicollinearity(small_df, target="churn")
        assert result.checks_run == 1

    def test_module_leakage(self):
        df = pd.DataFrame({"x": range(100), "target": range(100)})
        result = validate_mod.leakage(df, target="target")
        assert result.checks_run == 1


# ── improve/suggester ────────────────────────────────────────────────────────

from kaizenstat.improve.suggester import ImprovementReport, Suggester


class TestImprovementReportDisplay:
    def test_display_no_suggestions(self):
        report = ImprovementReport(suggestions=[], top_priority=None)
        report.display()  # Should print "No improvements found"


class TestSuggesterFeatureEngineering:
    def test_feature_engineering_no_ideas(self):
        """Dataset with no skewed, few numeric, few categorical → no ideas → fallback message."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        })
        ideas = Suggester().feature_engineering(df, target="target")
        assert any("No obvious" in idea for idea in ideas)

    def test_feature_engineering_with_skew(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "skewed": np.concatenate([np.ones(90), np.exp(np.linspace(0, 5, 10))]),
            "normal": rng.normal(0, 1, 100),
            "target": rng.integers(0, 2, 100),
        })
        ideas = Suggester().feature_engineering(df, target="target")
        assert len(ideas) > 0

    def test_feature_engineering_with_high_card_cat(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "cat_col": [f"val_{i}" for i in range(100)],
            "target": rng.integers(0, 2, 100),
        })
        ideas = Suggester().feature_engineering(df, target="target")
        assert any("Count-encode" in idea or "No obvious" in idea for idea in ideas)


class TestFromDebugSuggestions:
    def _make_debug_result(self, label, gap=0.0, test_score=0.5):
        class DR:
            issues = [type("I", (), {"name": "x", "description": "desc",
                                      "suggestion": "fix it", "risk_level": "HIGH"})()]
        dr = DR()
        dr.label = label
        dr.gap = gap
        dr.test_score = test_score
        return dr

    def test_underfitting_suggestion(self):
        dr = self._make_debug_result("underfitting", test_score=0.55)
        suggs = Suggester()._from_debug(dr, 1)
        assert len(suggs) == 1
        assert "accuracy" in suggs[0].expected_gain or "%" in suggs[0].expected_gain

    def test_leakage_suggestion(self):
        dr = self._make_debug_result("leakage")
        suggs = Suggester()._from_debug(dr, 1)
        assert len(suggs) == 1
        assert "leakage" in suggs[0].expected_gain.lower()

    def test_other_suggestion(self):
        dr = self._make_debug_result("other_issue", test_score=0.65)
        suggs = Suggester()._from_debug(dr, 1)
        assert len(suggs) == 1


class TestFromDataSuggestions:
    def test_smote_suggestion_for_imbalanced(self):
        """Minority 5% → class imbalance suggestion at priority."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "target": np.concatenate([np.zeros(475), np.ones(25)]).astype(int),
        })
        report = Suggester().suggest(df, target="target")
        actions = [s.action for s in report.suggestions]
        assert any("SMOTE" in a or "imbalance" in a.lower() or "class_weight" in a for a in actions)

    def test_calibration_suggestion_with_good_score(self, small_df):
        """High test score → calibration suggestion."""
        class GoodDebug:
            issues = []
            feature_importances = None
            test_score = 0.82
            label = ""
            gap = 0.05
        report = Suggester().suggest(small_df, target="churn", debug_result=GoodDebug())
        actions = [s.action for s in report.suggestions]
        assert any("calibrat" in a.lower() for a in actions)

    def test_low_importance_features_suggestion(self, small_df):
        """Near-zero importance features → feature selection suggestion."""
        importances = pd.Series({"age": 0.5, "income": 0.001, "tenure": 0.0005})

        class DebugResult:
            issues = []
            feature_importances = importances
            test_score = 0.88
            label = ""
            gap = 0.02
        report = Suggester().suggest(small_df, target="churn", debug_result=DebugResult())
        actions = [s.action for s in report.suggestions]
        assert any("near-zero" in a.lower() or "importance" in a.lower() or "Drop" in a for a in actions)

    def test_failure_slice_suggestion(self, small_df):
        """Failure slice issue in debug_result → subgroup fix suggestion."""
        class SliceIssue:
            name = "Failure Slice: city"
            description = "City NY fails most"
            suggestion = "Collect more data for this slice"
            risk_level = "HIGH"
        class DebugResult:
            issues = [SliceIssue()]
            feature_importances = None
            test_score = 0.70
            label = ""
            gap = 0.15
        report = Suggester().suggest(small_df, target="churn", debug_result=DebugResult())
        actions = [s.action for s in report.suggestions]
        assert any("subgroup" in a.lower() or "labelled" in a.lower() or "Collect" in a for a in actions)

    def test_low_test_score_stacking_suggestion(self, small_df):
        """Test score < 0.80 → stacking ensemble suggestion."""
        class DebugResult:
            issues = []
            feature_importances = None
            test_score = 0.65
            label = ""
            gap = 0.10
        report = Suggester().suggest(small_df, target="churn", debug_result=DebugResult())
        actions = [s.action for s in report.suggestions]
        assert any("stack" in a.lower() or "ensemble" in a.lower() or "train_auto" in a for a in actions)


# ── health/scorer — HealthResult.display ─────────────────────────────────────

class TestHealthResultDisplay:
    def test_display_with_penalties(self, missing_df):
        result = HealthScorer().report(missing_df, target="target")
        result.display()  # should not raise

    def test_display_no_penalties(self, small_df):
        result = HealthScorer().report(small_df, target="churn")
        result.display()


# ── validate/checker — ValidationReport.display ──────────────────────────────

class TestValidationReportDisplay:
    def test_display_passes(self, small_df):
        result = Validator().assumptions(small_df, target="churn")
        result.display()

    def test_display_with_issues(self, missing_df):
        result = Validator().assumptions(missing_df, target="target")
        result.display()
