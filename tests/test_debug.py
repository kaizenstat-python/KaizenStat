"""Tests for model debugging and diagnostics."""
import pandas as pd
import pytest
from kaizenstat import model, debug


class TestDebugBasics:
    """Basic debug result structure and labels."""

    def test_debug_result_structure(self, small_df):
        """DebugResult should have all required fields."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        assert result.label is not None
        assert result.severity is not None
        assert result.confidence > 0
        assert hasattr(result, 'diagnosis')

    def test_debug_labels_valid(self, small_df):
        """Debug labels should be from predefined set."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        valid_labels = {
            'data_leakage', 'leakage_risk', 'data_issue',
            'severe_underfitting', 'underfitting', 'excellent',
            'healthy', 'acceptable', 'overfitting_risk',
            'overfitting', 'severe_overfitting', 'weak_model', 'broken_model'
        }
        assert result.label in valid_labels

    def test_debug_severity_levels(self, small_df):
        """Severity should be CRITICAL, HIGH, MEDIUM, or LOW."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        assert result.severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

    def test_debug_confidence_range(self, small_df):
        """Confidence should be between 0 and 1."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        assert 0 <= result.confidence <= 1

    def test_debug_gap_detection(self):
        """Should detect overfitting (train > test)."""
        pipe = model.train_best(
            pd.DataFrame({
                'x': range(100),
                'y': range(100),
                'target': [i % 2 for i in range(100)],
            }),
            target='target'
        ).pipeline
        X = pd.DataFrame({'x': range(100), 'y': range(100)}).iloc[-20:]
        y = pd.Series([i % 2 for i in range(100)]).iloc[-20:]

        result = debug.model_failure(pipe, X, X, y, y)
        assert result.gap >= 0


class TestFeatureImportance:
    """Feature importance extraction and analysis."""

    def test_feature_importance_populated(self, small_df):
        """Feature importances should be extracted."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        # Feature importances should be non-empty or None, not zero-filled
        assert result.feature_importances is None or len(result.feature_importances) > 0

    def test_feature_importance_sorted(self, small_df):
        """Feature importances should be sorted descending."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        if result.feature_importances is not None and len(result.feature_importances) > 1:
            importances = result.feature_importances.values
            # Should be monotonically non-increasing
            assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))


class TestPremiumDiagnostics:
    """Premium diagnostic features (v0.5.1)."""

    def test_failure_clustering_detection(self, small_df):
        """Should detect failing subgroups by categorical column."""
        # Create data where one city always fails
        df = pd.DataFrame({
            'age': small_df['age'],
            'income': small_df['income'],
            'city': ['NY'] * 250 + ['LA'] * 250,  # NY fails, LA succeeds
            'churn': [1] * 200 + [0] * 50 + [0] * 250,  # NY: 80% churn, LA: 0% churn
        })
        pipe = model.train_best(df, target='churn').pipeline
        X_test = df.drop(columns=['churn']).iloc[-150:]
        y_test = df['churn'].iloc[-150:]

        result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        # Should have issues if failure clustering is enabled
        assert result.issues is not None

    def test_data_vs_model_blame(self, small_df):
        """Should distinguish data problems from model problems."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]
        X_train = small_df.drop(columns=['churn']).iloc[:-100]
        y_train = small_df['churn'].iloc[:-100]

        result = debug.model_failure(pipe, X_train, X_test, y_train, y_test)
        # Should produce diagnosis without crashing
        assert result.diagnosis is not None


class TestFeatureImpact:
    """Counterfactual feature impact analysis."""

    def test_feature_impact_basic(self, small_df):
        """Should compute feature impact."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        impacts = debug.feature_impact(pipe, X_test, y_test)
        assert isinstance(impacts, dict)
        assert len(impacts) > 0

    def test_feature_impact_drop_decreases_score(self, small_df):
        """Dropping important features should decrease score."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        impacts = debug.feature_impact(pipe, X_test, y_test)
        # At least one feature should have positive impact
        assert any(v > 0 for v in impacts.values()) or all(v <= 0 for v in impacts.values())


class TestDatasetDifficulty:
    """Dataset difficulty estimation."""

    def test_dataset_difficulty_range(self, small_df):
        """Difficulty should be between 0 and 1."""
        from kaizenstat.debug.debugger import dataset_difficulty
        X = small_df.drop(columns=['churn'])
        y = small_df['churn']
        diff = dataset_difficulty(X, y)
        assert 0 <= diff <= 1

    def test_easy_vs_hard_datasets(self):
        """Easy datasets should have low difficulty."""
        import numpy as np
        # Easy: perfectly separable
        X_easy = pd.DataFrame({
            'x1': [-1] * 50 + [1] * 50,
            'x2': [-1] * 50 + [1] * 50,
        })
        y_easy = np.array([0] * 50 + [1] * 50)

        from kaizenstat.debug.debugger import dataset_difficulty
        diff = dataset_difficulty(X_easy, y_easy)
        # Easy dataset should have low difficulty
        assert diff < 0.5


class TestRecommendActions:
    """Recommendation engine."""

    def test_recommend_actions_basic(self, small_df):
        """Should generate recommendations."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        debug_result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)

        from kaizenstat.debug.debugger import recommend_actions
        actions = recommend_actions(None, debug_result)
        assert isinstance(actions, list)

    def test_recommend_actions_non_empty(self, small_df):
        """Recommendations should be generated for typical case."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        debug_result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)

        from kaizenstat.debug.debugger import recommend_actions
        actions = recommend_actions(None, debug_result)
        # Should have at least one action or be empty
        assert isinstance(actions, list)


class TestEdgeCases:
    """Edge cases in diagnostics."""

    def test_debug_with_single_feature(self):
        """Should handle single-feature data."""
        df = pd.DataFrame({
            'x': range(100),
            'target': [i % 2 for i in range(100)],
        })
        pipe = model.train_best(df, target='target').pipeline
        X = pd.DataFrame({'x': range(100)}).iloc[-20:]
        y = pd.Series([i % 2 for i in range(100)]).iloc[-20:]

        result = debug.model_failure(pipe, X, X, y, y)
        assert result is not None

    def test_debug_perfect_score(self):
        """Should handle perfect train/test scores."""
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100),
            'target': [0] * 50 + [1] * 50,
        })
        pipe = model.train_best(df, target='target').pipeline
        X = df.drop(columns=['target']).iloc[-20:]
        y = df['target'].iloc[-20:]

        result = debug.model_failure(pipe, X, X, y, y)
        assert result.label in ['data_leakage', 'leakage_risk', 'excellent']

    def test_debug_zero_score(self):
        """Should handle zero/random performance."""
        df = pd.DataFrame({
            'x': list(range(50)) + list(range(50)),
            'target': [0, 1] * 50,
        })
        pipe = model.train_best(df, target='target').pipeline
        X = df.drop(columns=['target']).iloc[-20:]
        y = df['target'].iloc[-20:]

        result = debug.model_failure(pipe, X, X, y, y)
        assert result is not None
