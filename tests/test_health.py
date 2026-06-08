"""Tests for health scoring module."""
import pandas as pd
import pytest
from kaizenstat.health import scorer


class TestHealthScoring:
    """Health score computation and penalties."""

    def test_clean_dataset_high_score(self, small_df):
        """Clean dataset should score >= 80."""
        result = scorer.report(small_df, target='churn')
        assert result.score >= 80
        assert result.grade in ['A', 'B']
        assert len(result.penalties) == 0

    def test_missing_values_penalty(self, missing_df):
        """Missing values should trigger penalty and lower score."""
        result = scorer.report(missing_df, target='target')
        assert any(p.name == 'Missing Values' for p in result.penalties)
        assert result.score < 100

    def test_outlier_detection(self, outlier_df):
        """Extreme outliers should trigger penalty."""
        result = scorer.report(outlier_df, target='target')
        assert any(p.name == 'Outliers' for p in result.penalties)

    def test_skewness_detection(self, skewed_df):
        """Highly skewed features should trigger penalty."""
        result = scorer.report(skewed_df, target='target')
        assert any(p.name == 'High Skewness' for p in result.penalties)

    def test_class_imbalance_detection(self, imbalanced_df):
        """Class imbalance < 10% should trigger penalty."""
        result = scorer.report(imbalanced_df, target='target')
        assert any(p.name == 'Class Imbalance' for p in result.penalties)
        assert result.score < 100

    def test_duplicate_rows(self):
        """Duplicate rows should trigger penalty."""
        df = pd.DataFrame({
            'x': [1, 2, 1, 2],
            'y': [1, 1, 1, 1],
        })
        result = scorer.report(df, target='y')
        assert any(p.name == 'Duplicate Rows' for p in result.penalties)

    def test_constant_features(self):
        """Zero-variance columns should trigger penalty."""
        df = pd.DataFrame({
            'x': [1, 1, 1, 1],
            'y': [1, 2, 1, 2],
        })
        result = scorer.report(df, target='y')
        assert any(p.name == 'Constant Features' for p in result.penalties)

    def test_high_cardinality(self):
        """Categorical string column with > 50 unique values should trigger penalty."""
        df = pd.DataFrame({
            'high_card': [f'cat_{i}' for i in range(100)],  # 100 unique string values
            'target': [i % 2 for i in range(100)],
        })
        result = scorer.report(df, target='target')
        assert any(p.name == 'High Cardinality' for p in result.penalties)

    def test_leakage_detection(self):
        """Feature with corr > 0.98 to target should trigger leakage penalty."""
        df = pd.DataFrame({
            'leak': range(100),
            'target': range(100),  # Perfect correlation
        })
        result = scorer.report(df, target='target')
        assert any(p.name == 'Leakage Proxy' for p in result.penalties)

    def test_grade_assignment(self, small_df):
        """Grade should be A–F based on score."""
        result = scorer.report(small_df, target='churn')
        assert result.grade in ['A', 'B', 'C', 'D', 'F']

    def test_single_row_edge_case(self):
        """Single row dataset should not crash."""
        df = pd.DataFrame({'x': [1], 'target': [0]})
        result = scorer.report(df, target='target')
        assert result.score >= 0
        assert result.score <= 100

    def test_all_missing_column(self):
        """Column with all NaN should be detected."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'all_nan': [None, None, None],
            'target': [0, 1, 0],
        })
        result = scorer.report(df, target='target')
        # Should have a missing values penalty
        assert result.score < 100

    def test_score_scale_0_to_100(self, small_df):
        """Score should always be between 0 and 100."""
        result = scorer.report(small_df, target='churn')
        assert 0 <= result.score <= 100
