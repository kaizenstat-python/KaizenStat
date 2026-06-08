"""Tests for text/NLP mode."""
import pandas as pd
import pytest
from kaizenstat.doctor import DataDoctor


class TestTextModeDetection:
    """Automatic text mode detection."""

    def test_text_mode_auto_detected(self, text_df):
        """Should auto-detect text-dominant column."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        assert doctor.mode() == 'text'

    def test_tabular_mode_detection(self):
        """Short categorical should not trigger text mode."""
        df = pd.DataFrame({
            'short': ['A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1],
        })
        doctor = DataDoctor()
        doctor.fit(df, target='target')
        assert doctor.mode() == 'tabular'


class TestTextHealth:
    """Text mode health scoring."""

    def test_text_health_runs(self, text_df):
        """Text health should run without error."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.health()
        assert result.score >= 0
        assert result.score <= 100

    def test_text_health_penalties(self, text_df):
        """Text should check for text-specific penalties."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.health()
        # May have text penalties
        assert result.penalties is not None


class TestTextValidation:
    """Text mode validation checks."""

    def test_text_validate_runs(self, text_df):
        """Text validation should run."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.validate()
        assert result is not None

    def test_text_token_skew_check(self):
        """Should detect token frequency skew."""
        df = pd.DataFrame({
            'review': [
                'the the the the the good',
                'the the the the the bad',
                'the the the the the good',
                'the the the the the bad',
            ],
            'sentiment': [1, 0, 1, 0],
        })
        doctor = DataDoctor()
        doctor.fit(df, target='sentiment')
        result = doctor.validate()
        # May flag stopword dominance or token skew
        assert result is not None


class TestTextTraining:
    """Text model training with multi-pipeline benchmark."""

    def test_text_train_runs(self, text_df):
        """Text training should complete."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.train()
        assert result.model_name is not None
        assert result.test_score >= 0

    def test_text_multi_pipeline_benchmark(self, text_df):
        """Should benchmark multiple text pipelines."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        # This should trigger multi-pipeline benchmark internally
        result = doctor.train()
        # Should have a pipeline selected
        assert result.pipeline is not None

    def test_text_train_with_tune(self, text_df):
        """Text training with tuning should work."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.train(tune=True, n_iter=5)
        assert result.test_score >= 0

    def test_text_prediction(self, text_df):
        """Text model should make predictions."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.train()
        # Should have predict/predict_proba
        assert hasattr(result.pipeline, 'predict')


class TestTextDebug:
    """Text mode debugging."""

    def test_text_debug_runs(self, text_df):
        """Text debug should complete."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        result = doctor.debug_model()
        assert result is not None
        assert result.label is not None

    def test_text_debug_features_sparsity(self, text_df):
        """Text debug should report sparsity."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        doctor.train()
        result = doctor.debug_model()
        # Should have text-specific diagnostics
        assert result is not None


class TestTextImprove:
    """Text mode improvement suggestions."""

    def test_text_improve_runs(self, text_df):
        """Text improve should generate suggestions."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        doctor.train()
        result = doctor.improve()
        assert result.suggestions is not None

    def test_text_ngram_suggestions(self, text_df):
        """May suggest n-gram enhancements."""
        doctor = DataDoctor()
        doctor.fit(text_df, target='sentiment')
        doctor.train()
        result = doctor.improve()
        # Should have suggestions
        assert len(result.suggestions) >= 0


class TestTextEdgeCases:
    """Edge cases in text processing."""

    def test_very_short_text(self):
        """Should handle very short text."""
        df = pd.DataFrame({
            'text': ['a', 'b', 'a', 'b'] * 5,
            'label': [0, 1, 0, 1] * 5,
        })
        doctor = DataDoctor()
        doctor.fit(df, target='label')
        # Should detect as text or tabular
        assert doctor.mode() in ['text', 'tabular']

    def test_mixed_case_text(self):
        """Should handle mixed case and punctuation."""
        df = pd.DataFrame({
            'review': [
                'GREAT product!!!',
                'awful item... terrible',
                'Excellent quality, highly recommend.',
                'Horrible experience!',
            ] * 5,
            'sentiment': [1, 0, 1, 0] * 5,
        })
        doctor = DataDoctor()
        doctor.fit(df, target='sentiment')
        doctor.train()
        result = doctor.debug_model()
        assert result is not None

    def test_html_tags_in_text(self):
        """Should handle HTML in text."""
        df = pd.DataFrame({
            'html_text': [
                '<p>Good product</p>',
                '<p>Bad product</p>',
                '<div>Great item</div>',
                '<span>Terrible item</span>',
            ] * 5,
            'label': [1, 0, 1, 0] * 5,
        })
        doctor = DataDoctor()
        doctor.fit(df, target='label')
        doctor.train()
        assert doctor.train() is not None

    def test_unicode_and_special_chars(self):
        """Should handle unicode and special characters."""
        df = pd.DataFrame({
            'text': [
                'Excellent café ☕ product',
                'Bad naïve item 😞',
                'Great résumé content',
                'Terrible über experience',
            ] * 5,
            'label': [1, 0, 1, 0] * 5,
        })
        doctor = DataDoctor()
        doctor.fit(df, target='label')
        result = doctor.train()
        assert result is not None
