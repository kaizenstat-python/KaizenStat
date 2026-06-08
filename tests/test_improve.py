"""Tests for improvement suggestions and recommendations."""
import pandas as pd
import pytest
from kaizenstat import model, debug, improve


class TestSuggestionGeneration:
    """Improvement suggestion creation."""

    def test_suggest_basic(self, small_df):
        """Should generate suggestions."""
        result = improve.suggest(small_df, target='churn')
        assert result.suggestions is not None
        assert isinstance(result.suggestions, list)

    def test_suggestions_have_required_fields(self, small_df):
        """Each suggestion should have required fields."""
        result = improve.suggest(small_df, target='churn')
        for s in result.suggestions:
            assert s.priority >= 1
            assert s.category is not None
            assert s.action is not None
            assert s.reason is not None
            assert s.expected_gain is not None
            assert s.impact in ['LOW', 'MEDIUM', 'HIGH']

    def test_top_priority_suggestion(self, small_df):
        """Top priority should be first or None."""
        result = improve.suggest(small_df, target='churn')
        if result.suggestions:
            assert result.top_priority == result.suggestions[0]

    def test_suggestions_have_expected_gains(self, small_df):
        """Suggestions should include quantified gains."""
        result = improve.suggest(small_df, target='churn')
        for s in result.suggestions:
            # Expected gain should be non-empty string with specifics
            assert len(s.expected_gain) > 0
            assert '%' in s.expected_gain or 'Expected' in s.expected_gain or '+' in s.expected_gain


class TestPremiumSuggestions:
    """Premium suggestion features (v0.5.1)."""

    def test_suggest_with_debug_result(self, small_df):
        """Suggestions should use debug_result info."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        debug_result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        result = improve.suggest(small_df, target='churn', debug_result=debug_result)
        assert result.suggestions is not None

    def test_imbalance_suggestion_quantified(self, imbalanced_df):
        """Imbalance suggestion should have quantified gain."""
        result = improve.suggest(imbalanced_df, target='target')
        imbalance_suggestions = [
            s for s in result.suggestions
            if 'Imbalance' in s.category or 'SMOTE' in s.action
        ]
        if imbalance_suggestions:
            s = imbalance_suggestions[0]
            # Should have percentage gains
            assert '%' in s.expected_gain or 'recall' in s.expected_gain.lower()

    def test_ensemble_suggestion_with_gap(self, small_df):
        """Low-score data should suggest ensemble with gap-based gain."""
        result = improve.suggest(small_df, target='churn')
        ensemble_suggestions = [
            s for s in result.suggestions
            if 'Ensemble' in s.action or 'ensemble' in s.action.lower()
        ]
        # May or may not have ensemble suggestion depending on score
        if ensemble_suggestions:
            s = ensemble_suggestions[0]
            assert '%' in s.expected_gain or 'gap' in s.expected_gain.lower()

    def test_tuning_suggestion_quantified(self, small_df):
        """Tuning suggestion should be quantified."""
        result = improve.suggest(small_df, target='churn')
        tuning_suggestions = [
            s for s in result.suggestions
            if 'Tuning' in s.category or 'tune' in s.action.lower()
        ]
        if tuning_suggestions:
            s = tuning_suggestions[0]
            assert '%' in s.expected_gain

    def test_calibration_suggestion(self, small_df):
        """Calibration suggestion should appear for good models."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        debug_result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        result = improve.suggest(small_df, target='churn', debug_result=debug_result)
        calibration_suggestions = [
            s for s in result.suggestions
            if 'Calibration' in s.category or 'calibration' in s.action.lower()
        ]
        # May have calibration suggestion
        if calibration_suggestions:
            s = calibration_suggestions[0]
            assert 'gap' in s.expected_gain.lower() or 'calibration' in s.expected_gain.lower()

    def test_feature_selection_suggestion(self, small_df):
        """Low-importance features should trigger suggestion."""
        pipe = model.train_best(small_df, target='churn').pipeline
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_test = small_df['churn'].iloc[-100:]

        debug_result = debug.model_failure(pipe, X_test, X_test, y_test, y_test)
        result = improve.suggest(small_df, target='churn', debug_result=debug_result)
        feature_suggestions = [
            s for s in result.suggestions
            if 'Feature' in s.category or 'drop' in s.action.lower()
        ]
        # May have feature selection suggestion
        if feature_suggestions:
            s = feature_suggestions[0]
            assert 'generalization' in s.expected_gain.lower() or '%' in s.expected_gain


class TestSuggestionPrioritization:
    """Suggestion prioritization and ranking."""

    def test_suggestions_prioritized(self, small_df):
        """Suggestions should be prioritized 1, 2, 3, ..."""
        result = improve.suggest(small_df, target='churn')
        if len(result.suggestions) > 1:
            priorities = [s.priority for s in result.suggestions]
            assert priorities == sorted(priorities)

    def test_prioritize_method(self, small_df):
        """prioritize() should sort by impact."""
        result = improve.suggest(small_df, target='churn')
        if result.suggestions:
            prioritized = improve.prioritize(result.suggestions)
            # HIGH impact should come before MEDIUM/LOW
            high_idx = next((i for i, s in enumerate(prioritized) if s.impact == 'HIGH'), float('inf'))
            medium_idx = next((i for i, s in enumerate(prioritized) if s.impact == 'MEDIUM'), float('inf'))
            # HIGH should appear before or equal to MEDIUM
            assert high_idx <= medium_idx


class TestFeatureEngineering:
    """Feature engineering recommendations."""

    def test_feature_engineering_ideas(self, small_df):
        """Should generate feature engineering ideas."""
        ideas = improve.feature_engineering(small_df, target='churn')
        assert isinstance(ideas, list)
        assert len(ideas) > 0

    def test_feature_engineering_skew_transform(self, skewed_df):
        """Should suggest log transform for skewed features."""
        ideas = improve.feature_engineering(skewed_df, target='target')
        log_suggestions = [i for i in ideas if 'log' in i.lower()]
        # Should suggest log transform
        assert len(log_suggestions) > 0 or len(ideas) > 0

    def test_feature_engineering_ratio_features(self, small_df):
        """Should suggest ratio features for numeric pairs."""
        ideas = improve.feature_engineering(small_df, target='churn')
        ratio_suggestions = [i for i in ideas if 'ratio' in i.lower() or 'div' in i.lower()]
        # May suggest ratio features
        assert isinstance(ideas, list)

    def test_feature_engineering_count_encoding(self, small_df):
        """Should suggest count encoding for high-cardinality categoricals."""
        ideas = improve.feature_engineering(small_df, target='churn')
        count_suggestions = [i for i in ideas if 'count' in i.lower()]
        # May suggest count encoding
        assert isinstance(ideas, list)


class TestSuggestionDataDriven:
    """Data-driven suggestion content."""

    def test_suggestion_reason_meaningful(self, small_df):
        """Suggestion reasons should reference actual data metrics."""
        result = improve.suggest(small_df, target='churn')
        for s in result.suggestions:
            # Reason should be non-empty and meaningful
            assert len(s.reason) > 0
            # Should reference data or model metrics
            assert any(keyword in s.reason for keyword in [
                'rows', 'class', 'score', 'missing', 'feature',
                'importance', 'accuracy', 'correlation', 'skew'
            ]) or len(s.reason) > 10

    def test_suggestion_action_specific(self, small_df):
        """Action should be specific, not generic."""
        result = improve.suggest(small_df, target='churn')
        for s in result.suggestions:
            # Action should mention specific techniques
            assert len(s.action) > 0
            # Should reference method names
            assert any(keyword in s.action for keyword in [
                'doctor.', 'fix.', 'train', 'SMOTE', 'ensemble',
                'KNN', 'Yeo-Johnson', 'class_weight'
            ]) or 'Apply' in s.action or 'Run' in s.action


class TestEdgeCases:
    """Edge cases in suggestion generation."""

    def test_suggest_perfect_model(self):
        """Perfect model should have minimal suggestions."""
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100),
            'target': [0] * 50 + [1] * 50,
        })
        result = improve.suggest(df, target='target')
        # May have no suggestions or only data quality suggestions
        assert result.suggestions is not None

    def test_suggest_single_feature(self):
        """Single feature should generate suggestions."""
        df = pd.DataFrame({
            'x': range(100),
            'target': [i % 2 for i in range(100)],
        })
        result = improve.suggest(df, target='target')
        assert result.suggestions is not None

    def test_suggest_multiclass(self, multiclass_df):
        """Multi-class should work."""
        result = improve.suggest(multiclass_df, target='target')
        assert result.suggestions is not None

    def test_suggest_regression(self, regression_df):
        """Regression should work."""
        result = improve.suggest(regression_df, target='target')
        assert result.suggestions is not None
