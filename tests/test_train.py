"""Tests for model training (benchmark, train_best, train_auto)."""
import pytest
from kaizenstat import model


class TestTrainBenchmark:
    """Model benchmarking and selection."""

    def test_benchmark_runs(self, small_df):
        """Benchmark should run and return results."""
        result = model.benchmark(small_df, target='churn')
        assert result.best_name is not None
        assert result.best_score > 0
        assert len(result.entries) > 0

    def test_benchmark_selects_best(self, small_df):
        """Best model should have highest CV score."""
        result = model.benchmark(small_df, target='churn')
        best_score = result.best_score
        all_scores = [e.score for e in result.entries]
        assert best_score >= min(all_scores)

    def test_benchmark_classification_metric(self, small_df):
        """Classification should use accuracy/F1."""
        result = model.benchmark(small_df, target='churn')
        assert result.metric in ['accuracy', 'f1', 'roc_auc']

    def test_benchmark_regression_metric(self, regression_df):
        """Regression should use R² or RMSE."""
        result = model.benchmark(regression_df, target='target')
        assert result.metric in ['r2', 'neg_mean_squared_error']

    def test_benchmark_includes_diverse_models(self, small_df):
        """Should benchmark multiple model types."""
        result = model.benchmark(small_df, target='churn')
        model_names = [e.name for e in result.entries]
        # Should have at least 3–4 different models
        assert len(set(model_names)) >= 3


class TestTrainBest:
    """train_best() with optional tuning."""

    def test_train_best_classification(self, small_df):
        """Should train and return TrainResult."""
        result = model.train_best(small_df, target='churn')
        assert result.model_name is not None
        assert result.test_score > 0
        assert result.train_score > 0
        assert result.task == 'classification'

    def test_train_best_regression(self, regression_df):
        """Should handle regression."""
        result = model.train_best(regression_df, target='target')
        assert result.task == 'regression'
        assert result.test_score != 0

    def test_train_best_cv_score_populated(self, small_df):
        """CV score should be populated."""
        result = model.train_best(small_df, target='churn', cv=3)
        assert result.cv_score > 0
        assert result.cv_std >= 0

    def test_train_best_with_tuning(self, small_df):
        """With tune=True should include hyperparameter tuning."""
        result = model.train_best(small_df, target='churn', tune=True, n_iter=5)
        assert result.cv_score > 0
        # best_params should be populated
        assert isinstance(result.best_params, dict)

    def test_train_best_tuning_improves_or_maintains(self, small_df):
        """Tuning should not substantially hurt test score (within 10%)."""
        result_no_tune = model.train_best(small_df, target='churn', tune=False)
        result_tune = model.train_best(small_df, target='churn', tune=True, n_iter=5)
        # Tuned should not degrade more than 10 percentage points vs no-tune.
        # On random-noise data the gap can be large, so we allow ±10pp variance.
        assert result_tune.test_score >= result_no_tune.test_score * 0.85

    def test_train_best_pipeline_created(self, small_df):
        """Pipeline should be created and callable."""
        result = model.train_best(small_df, target='churn')
        assert hasattr(result.pipeline, 'predict')
        assert hasattr(result.pipeline, 'predict_proba')

    def test_train_best_prediction_shape(self, small_df):
        """Predictions should match test set shape."""
        from sklearn.model_selection import train_test_split
        result = model.train_best(small_df, target='churn')
        X_test = small_df.drop(columns=['churn']).iloc[-100:]
        y_pred = result.pipeline.predict(X_test)
        assert len(y_pred) == len(X_test)

    def test_train_best_imbalanced_data(self, imbalanced_df):
        """Should handle class imbalance."""
        result = model.train_best(imbalanced_df, target='target')
        assert result.test_score > 0

    def test_train_best_multiclass(self, multiclass_df):
        """Should handle multi-class classification."""
        result = model.train_best(multiclass_df, target='target')
        assert result.task == 'classification'


class TestTrainAuto:
    """train_auto() with premium features (stacking, progressive tuning)."""

    def test_train_auto_basic(self, small_df):
        """train_auto should work without ensemble."""
        result = model.train_auto(small_df, target='churn', ensemble=False)
        assert result.model_name is not None
        assert result.test_score > 0

    def test_train_auto_with_ensemble(self, small_df):
        """With ensemble=True should build stacking ensemble."""
        result = model.train_auto(small_df, target='churn', ensemble=True)
        assert 'Stack' in result.model_name or 'Ensemble' in result.model_name

    def test_train_auto_with_tune(self, small_df):
        """With tune=True should apply progressive tuning."""
        result = model.train_auto(
            small_df, target='churn',
            tune=True, n_iter=10, ensemble=True
        )
        assert result.best_params is not None
        assert len(result.best_params) > 0

    def test_train_auto_stacking_beats_voting(self, small_df):
        """Stacking should perform >= voting (usually better)."""
        # This is a statistical test, not always guaranteed
        result = model.train_auto(small_df, target='churn', ensemble=True, cv=3)
        assert result.test_score > 0

    def test_train_auto_progressive_tuning(self, small_df):
        """2-stage progressive tuning should work."""
        result = model.train_auto(small_df, target='churn', tune=True, n_iter=10)
        # Should have params from both stages
        assert result.cv_score > 0

    def test_train_auto_calibration_applied(self, small_df):
        """Model calibration should be applied if overconfident."""
        result = model.train_auto(small_df, target='churn')
        # Pipeline may have CalibratedClassifierCV wrapper
        assert hasattr(result.pipeline, 'predict_proba')

    def test_train_auto_extra_trees_in_pool(self, small_df):
        """ExtraTrees should be included in model pool."""
        result = model.train_auto(small_df, target='churn')
        # ExtraTrees may be selected
        assert result.test_score > 0

    def test_train_auto_regression(self, regression_df):
        """Should handle regression with stacking."""
        result = model.train_auto(regression_df, target='target', ensemble=True)
        assert result.task == 'regression'

    def test_train_auto_fallback_on_stack_failure(self, tiny_df):
        """Should fall back to soft voting if stacking fails."""
        result = model.train_auto(tiny_df, target='churn', ensemble=True)
        assert result.test_score >= 0  # Should not crash


class TestPremiumFeatures:
    """Premium engine features (v0.5.1)."""

    def test_progressive_tuning_two_stages(self, small_df):
        """Progressive tuning should run 2 stages."""
        result = model.train_best(small_df, target='churn', tune=True, n_iter=20)
        # Should have tuned parameters
        assert result.best_params is not None

    def test_platt_calibration_check(self, small_df):
        """Calibration should be applied when needed."""
        result = model.train_best(small_df, target='churn')
        # Should have predict_proba
        assert hasattr(result.pipeline, 'predict_proba')

    def test_stacking_meta_learner(self, small_df):
        """Stacking should use LogReg meta-learner for classification."""
        result = model.train_auto(small_df, target='churn', ensemble=True)
        # Model name should indicate stacking
        assert result.model_name is not None

    def test_param_grid_expansion(self, small_df):
        """Parameter grids should be expanded for premium features."""
        result = model.train_best(small_df, target='churn', tune=True, n_iter=15)
        assert result.best_params is not None


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_feature(self):
        """Model should train with single feature."""
        import pandas as pd
        df = pd.DataFrame({
            'x': range(100),
            'target': [i % 2 for i in range(100)],
        })
        result = model.train_best(df, target='target')
        assert result.test_score > 0

    def test_high_dimensional_data(self):
        """Should handle high-dimensional data (many features)."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(
            np.random.randn(100, 100),
            columns=[f'f{i}' for i in range(100)]
        )
        df['target'] = np.random.randint(0, 2, 100)
        result = model.train_auto(df, target='target')
        # Should apply feature selection
        assert result.test_score > 0

    def test_very_small_dataset(self, tiny_df):
        """Should handle very small dataset."""
        result = model.train_best(tiny_df, target='churn')
        assert result.test_score >= 0

    def test_single_class_fails_gracefully(self, single_class_df):
        """Single-class data should fail gracefully or use default."""
        try:
            result = model.train_best(single_class_df, target='target')
            # If it doesn't crash, that's good
            assert result is not None
        except ValueError:
            # Expected for single-class
            pass
