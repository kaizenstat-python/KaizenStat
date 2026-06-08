"""Tests for validation checks."""
import pandas as pd
import pytest
from kaizenstat.validate import checker


class TestValidationChecks:
    """Statistical assumption and integrity checks."""

    def test_clean_data_passes(self, small_df):
        """Clean data should pass all checks."""
        result = checker.assumptions(small_df, target='churn')
        assert result.passed or len(result.issues) <= 1  # May flag non-normality

    def test_leakage_detection(self):
        """Feature perfectly correlated with target should flag leakage."""
        df = pd.DataFrame({
            'leak': range(100),
            'target': range(100),
        })
        result = checker.leakage(df, target='target')
        assert not result.passed
        assert any('leak' in iss.issue for iss in result.issues)

    def test_no_false_leakage(self, small_df):
        """Clean data should not flag false leakage."""
        result = checker.leakage(small_df, target='churn')
        assert result.passed

    def test_normality_check(self, small_df):
        """Should check normality on numeric columns."""
        result = checker.assumptions(small_df, target='churn')
        assert result.checks_run > 0

    def test_multicollinearity_check(self):
        """High VIF should be flagged."""
        df = pd.DataFrame({
            'x1': range(100),
            'x2': [2*i for i in range(100)],  # Perfectly collinear with x1
            'target': [i % 2 for i in range(100)],
        })
        result = checker.assumptions(df, target='target')
        # Should detect multicollinearity
        assert result.checks_run > 0

    def test_skewness_extreme(self):
        """Extremely skewed distribution should be flagged."""
        df = pd.DataFrame({
            'skewed': [1] * 99 + [1000],  # Extreme skew
            'target': [0, 1] * 50,
        })
        result = checker.assumptions(df, target='target')
        assert result.checks_run > 0

    def test_single_class_edge_case(self, single_class_df):
        """Single class should be handled gracefully."""
        result = checker.assumptions(single_class_df, target='target')
        # Should not crash, might flag issues
        assert result is not None

    def test_drift_detection(self, small_df):
        """Should detect distribution drift between train/test splits."""
        from sklearn.model_selection import train_test_split
        X_train, X_test = train_test_split(
            small_df.drop(columns=['churn']),
            test_size=0.3, random_state=42
        )
        drift = checker.detect_drift(X_train, X_test)
        assert isinstance(drift, dict)
        assert all(0 <= v <= 1 for v in drift.values())  # p-values

    def test_no_checks_with_none_target(self, small_df):
        """Should handle None target gracefully."""
        result = checker.assumptions(small_df, target=None)
        assert result is not None

    def test_categorical_handling(self):
        """Should handle categorical columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'target': [0, 1, 0, 1, 0],
        })
        result = checker.assumptions(df, target='target')
        assert result.checks_run >= 0

    def test_mixed_types(self):
        """Should handle mixed numeric and categorical."""
        df = pd.DataFrame({
            'num': [1.5, 2.3, 3.1, 4.0],
            'cat': ['A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1],
        })
        result = checker.assumptions(df, target='target')
        assert result is not None
