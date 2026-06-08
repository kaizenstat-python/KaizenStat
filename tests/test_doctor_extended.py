"""Extended DataDoctor tests — uncovered branches and methods."""
import numpy as np
import pandas as pd
import pytest

from kaizenstat.doctor.data_doctor import ComparisonResult, DataDoctor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def doc_fitted(small_df):
    doc = DataDoctor()
    doc.fit(small_df, target="churn")
    return doc


@pytest.fixture
def doc_trained(small_df):
    doc = DataDoctor()
    doc.fit(small_df, target="churn")
    doc.train(cv=3)
    return doc


@pytest.fixture
def doc_full(small_df):
    doc = DataDoctor()
    doc.fit(small_df, target="churn")
    doc.health()
    doc.validate()
    doc.train(cv=3)
    doc.debug_model()
    doc.improve()
    return doc


@pytest.fixture
def regression_doc(regression_df):
    doc = DataDoctor()
    doc.fit(regression_df, target="target")
    return doc


# ── fit / mode / repr ─────────────────────────────────────────────────────────

class TestFitAndMode:
    def test_fit_returns_self(self, small_df):
        doc = DataDoctor()
        result = doc.fit(small_df, target="churn")
        assert result is doc

    def test_mode_tabular(self, doc_fitted):
        assert doc_fitted.mode() == "tabular"

    def test_repr_fitted(self, doc_fitted):
        r = repr(doc_fitted)
        assert "fitted=True" in r
        assert "churn" in r

    def test_repr_unfitted(self):
        doc = DataDoctor()
        r = repr(doc)
        assert "fitted=False" in r

    def test_require_fit_raises(self):
        doc = DataDoctor()
        with pytest.raises(RuntimeError, match="fit"):
            doc.health()

    def test_mode_requires_fit(self):
        doc = DataDoctor()
        with pytest.raises(RuntimeError):
            doc.mode()

    def test_fit_no_target(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)  # no target
        assert doc._target is None


# ── health / validate / fix ───────────────────────────────────────────────────

class TestHealthValidateFix:
    def test_health_returns_result(self, doc_fitted):
        result = doc_fitted.health()
        assert result.score >= 0

    def test_validate_returns_result(self, doc_fitted):
        result = doc_fitted.validate()
        assert result.checks_run > 0

    def test_fix_preview_returns_original(self, doc_fitted, small_df):
        result = doc_fitted.fix(preview_only=True)
        assert result.shape == small_df.shape

    def test_fix_applies_corrections(self, doc_fitted):
        result = doc_fitted.fix(safe=True)
        assert result is not None

    def test_fix_after_fit_uses_active_df(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")
        fixed = doc.fix(safe=True)
        assert isinstance(fixed, pd.DataFrame)


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrain:
    def test_train_classification(self, doc_fitted):
        result = doc_fitted.train(cv=3)
        assert result.task == "classification"

    def test_train_regression(self, regression_doc):
        result = regression_doc.train(cv=3)
        assert result.task == "regression"

    def test_train_no_target_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)  # no target
        with pytest.raises(ValueError, match="target"):
            doc.train()

    def test_train_with_tune(self, doc_fitted):
        result = doc_fitted.train(cv=3, tune=True, n_iter=5)
        assert result is not None


# ── debug_model ───────────────────────────────────────────────────────────────

class TestDebugModel:
    def test_debug_model_runs(self, doc_trained):
        result = doc_trained.debug_model()
        assert result is not None

    def test_debug_model_auto_trains(self, doc_fitted):
        result = doc_fitted.debug_model()
        assert result is not None
        assert doc_fitted._train_result is not None

    def test_debug_model_no_target_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)
        with pytest.raises(ValueError, match="target"):
            doc.debug_model()


# ── improve ───────────────────────────────────────────────────────────────────

class TestImprove:
    def test_improve_returns_report(self, doc_full):
        result = doc_full.improve()
        assert result is not None
        assert len(result.suggestions) >= 0

    def test_improve_without_prior_stages(self, doc_fitted):
        result = doc_fitted.improve()
        assert result is not None


# ── report ────────────────────────────────────────────────────────────────────

class TestReport:
    def test_report_generates_html(self, doc_full, tmp_path):
        path = str(tmp_path / "report.html")
        out = doc_full.report(output_path=path)
        import os
        assert os.path.exists(out)

    def test_report_with_no_results(self, doc_fitted, tmp_path):
        path = str(tmp_path / "empty_report.html")
        out = doc_fitted.report(output_path=path)
        import os
        assert os.path.exists(out)


# ── trust_score ───────────────────────────────────────────────────────────────

class TestTrustScore:
    def test_trust_score_after_debug(self, doc_full):
        result = doc_full.trust_score()
        assert 0 <= result.trust_score <= 100

    def test_trust_score_auto_trains(self, doc_fitted):
        result = doc_fitted.trust_score()
        assert 0 <= result.trust_score <= 100

    def test_trust_score_without_debug_split(self, doc_trained):
        result = doc_trained.trust_score()
        assert 0 <= result.trust_score <= 100

    def test_trust_score_regression(self, regression_doc):
        regression_doc.train(cv=3)
        result = regression_doc.trust_score()
        assert 0 <= result.trust_score <= 100


# ── train_auto ────────────────────────────────────────────────────────────────

class TestTrainAuto:
    def test_train_auto_runs(self, doc_fitted):
        result = doc_fitted.train_auto(cv=3, ensemble=True)
        assert result is not None

    def test_train_auto_no_target_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)
        with pytest.raises(ValueError, match="target"):
            doc.train_auto()


# ── detect_drift ──────────────────────────────────────────────────────────────

class TestDetectDrift:
    def test_detect_drift(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")
        X_train = small_df.drop(columns=["churn"]).iloc[:400]
        X_test = small_df.drop(columns=["churn"]).iloc[400:]
        result = doc.detect_drift(X_train, X_test)
        assert isinstance(result, dict)


# ── dataset_difficulty ────────────────────────────────────────────────────────

class TestDatasetDifficulty:
    def test_dataset_difficulty_range(self, doc_fitted):
        diff = doc_fitted.dataset_difficulty()
        assert 0.0 <= diff <= 1.0

    def test_dataset_difficulty_no_target_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)
        with pytest.raises(ValueError, match="target"):
            doc.dataset_difficulty()


# ── feature_impact ────────────────────────────────────────────────────────────

class TestFeatureImpact:
    def test_feature_impact_returns_dict(self, doc_trained):
        result = doc_trained.feature_impact(top_n=3)
        assert isinstance(result, dict)

    def test_feature_impact_no_model_raises(self, doc_fitted):
        with pytest.raises(RuntimeError, match="trained model"):
            doc_fitted.feature_impact()

    def test_feature_impact_no_target_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)
        doc._train_result = object()  # mock something
        with pytest.raises((ValueError, AttributeError)):
            doc.feature_impact()


# ── recommend_actions ─────────────────────────────────────────────────────────

class TestRecommendActions:
    def test_recommend_actions_returns_list(self, doc_full):
        result = doc_full.recommend_actions()
        assert isinstance(result, list)

    def test_recommend_actions_no_debug_raises(self, doc_trained):
        with pytest.raises(RuntimeError, match="debug"):
            doc_trained.recommend_actions()

    def test_recommend_actions_no_target_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df)
        doc._debug_result = object()
        with pytest.raises((ValueError, AttributeError)):
            doc.recommend_actions()


# ── pipeline_confidence ───────────────────────────────────────────────────────

class TestPipelineConfidence:
    def test_pipeline_confidence_range(self, doc_full):
        score = doc_full.pipeline_confidence()
        assert 0 <= score <= 100

    def test_pipeline_confidence_no_results(self, doc_fitted):
        score = doc_fitted.pipeline_confidence()
        assert 0 <= score <= 100

    def test_pipeline_confidence_all_stages(self, doc_full):
        score = doc_full.pipeline_confidence()
        assert isinstance(score, int)


# ── auto_improve ──────────────────────────────────────────────────────────────

class TestAutoImprove:
    def test_auto_improve_returns_comparison(self, doc_fitted):
        result = doc_fitted.auto_improve(tune=False)
        assert isinstance(result, ComparisonResult)

    def test_auto_improve_with_existing_train_result(self, doc_trained):
        result = doc_trained.auto_improve(tune=False)
        assert isinstance(result, ComparisonResult)

    def test_comparison_result_score_delta(self):
        from kaizenstat.model.trainer import TrainResult
        before = TrainResult(
            model_name="A", task="classification",
            train_score=0.9, test_score=0.8,
            cv_score=0.78, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        after = TrainResult(
            model_name="B", task="classification",
            train_score=0.91, test_score=0.85,
            cv_score=0.83, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        cr = ComparisonResult(before=before, after=after)
        assert abs(cr.score_delta - 0.05) < 0.001

    def test_comparison_result_display_improved(self):
        from kaizenstat.model.trainer import TrainResult
        before = TrainResult(
            model_name="A", task="classification",
            train_score=0.9, test_score=0.8,
            cv_score=0.78, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        after = TrainResult(
            model_name="B", task="classification",
            train_score=0.91, test_score=0.85,
            cv_score=0.83, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        ComparisonResult(before=before, after=after).display()

    def test_comparison_result_display_no_change(self):
        from kaizenstat.model.trainer import TrainResult
        tr = TrainResult(
            model_name="A", task="classification",
            train_score=0.9, test_score=0.8,
            cv_score=0.78, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        ComparisonResult(before=tr, after=tr).display()

    def test_comparison_result_display_regressed(self):
        from kaizenstat.model.trainer import TrainResult
        before = TrainResult(
            model_name="A", task="classification",
            train_score=0.9, test_score=0.85,
            cv_score=0.83, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        after = TrainResult(
            model_name="B", task="classification",
            train_score=0.88, test_score=0.80,
            cv_score=0.78, cv_std=0.02, best_params={},
            metrics={}, pipeline=None, label_encoder=None,
            feature_names=[],
        )
        ComparisonResult(before=before, after=after).display()


# ── add_model / add_check ─────────────────────────────────────────────────────

class TestPluginAPI:
    def test_add_model(self, small_df):
        from sklearn.svm import SVC
        doc = DataDoctor()
        doc.fit(small_df, target="churn")
        result = doc.add_model("MySVM", SVC(probability=True))
        assert result is doc
        assert "MySVM" in doc._custom_models

    def test_add_check_runs_during_validate(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")
        issues_found = []

        def my_check(df, target):
            issues_found.append("ran")
            return []

        doc.add_check(my_check, name="my_check")
        doc.validate()
        assert "ran" in issues_found

    def test_add_check_with_issues_reported(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")

        def bad_check(df, target):
            return ["Found a problem"]

        doc.add_check(bad_check, name="bad_check")
        doc.validate()

    def test_add_check_that_raises(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")

        def crashing_check(df, target):
            raise RuntimeError("check failed")

        doc.add_check(crashing_check, name="crash_check")
        result = doc.validate()  # Should not raise, just log the error
        assert result is not None

    def test_add_check_no_name_uses_function_name(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")

        def my_named_check(df, target):
            return []

        doc.add_check(my_named_check)
        assert any("my_named_check" in label for label, _ in doc._custom_checks)


# ── export_model / codegen ────────────────────────────────────────────────────

class TestExportCodegen:
    def test_export_model(self, doc_trained, tmp_path):
        path = str(tmp_path / "model.joblib")
        out = doc_trained.export_model(path=path)
        import os
        assert os.path.exists(out)

    def test_export_model_no_train_raises(self, doc_fitted):
        with pytest.raises(RuntimeError, match="train"):
            doc_fitted.export_model()

    def test_codegen_with_train(self, doc_trained, tmp_path):
        path = str(tmp_path / "pipeline.py")
        out = doc_trained.codegen(output_path=path)
        import os
        assert os.path.exists(out)

    def test_codegen_without_train(self, doc_fitted, tmp_path):
        path = str(tmp_path / "pipeline2.py")
        out = doc_fitted.codegen(output_path=path)
        import os
        assert os.path.exists(out)
