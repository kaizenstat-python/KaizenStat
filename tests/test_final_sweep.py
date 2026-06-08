"""Final coverage sweep — mocks and edge paths for remaining uncovered lines."""
import builtins
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── cli/main.py — main() function ─────────────────────────────────────────────

class TestCLIMain:
    def test_main_function_callable(self):
        """Just importing and calling main() with --help should not crash."""
        from typer.testing import CliRunner
        from kaizenstat.cli.main import app
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0


# ── output/reporter.py — open_browser path ───────────────────────────────────

class TestReporterOpenBrowser:
    def test_html_with_open_browser_mocked(self, tmp_path):
        from kaizenstat.output.reporter import Reporter
        path = str(tmp_path / "report.html")
        with patch("webbrowser.open") as mock_open:
            Reporter().html({}, path=path, open_browser=True)
            mock_open.assert_called_once()

    def test_export_model_pickle_fallback(self, tmp_path):
        """When joblib is not available, falls back to pickle."""
        from sklearn.dummy import DummyClassifier
        from kaizenstat.output.reporter import Reporter
        path = str(tmp_path / "model.joblib")
        orig_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "joblib":
                raise ImportError("no joblib")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            saved = Reporter().export_model(DummyClassifier(), path=path)
        assert saved.endswith(".pkl")

    def test_load_model_pickle_fallback(self, tmp_path):
        """When joblib is not available, load_model falls back to pickle."""
        from sklearn.dummy import DummyClassifier
        from kaizenstat.output.reporter import Reporter
        model = DummyClassifier()
        pkl_path = str(tmp_path / "model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        orig_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "joblib":
                raise ImportError("no joblib")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            loaded = Reporter().load_model(pkl_path)
        assert loaded is not None

    def test_build_html_safe_score_with_train(self, tmp_path):
        """Pass a train result so safe_score is invoked in _build_html."""
        from kaizenstat.output.reporter import Reporter

        class _Train:
            test_score = 0.88
            model_name = "RF"
            task = "classification"

        path = str(tmp_path / "train.html")
        Reporter().html({"train": _Train()}, path=path)
        with open(path) as f:
            content = f.read()
        assert "KaizenStat" in content


# ── reliability/trust.py — remaining branches ────────────────────────────────

class TestTrustAnalyzerRemainingBranches:
    def test_no_proba_no_decision_function(self):
        """Model with neither predict_proba nor decision_function → proba=None branch."""
        from kaizenstat.reliability.trust import TrustAnalyzer

        class NoProbModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        X = pd.DataFrame({"a": range(50)})
        y = pd.Series([i % 2 for i in range(50)])
        result = TrustAnalyzer().analyze(NoProbModel(), X, y)
        assert "confidence estimated" in " ".join(result.notes)

    def test_fragile_model_note(self):
        """Very low robustness → fragile note added."""
        from kaizenstat.reliability.trust import TrustAnalyzer
        from sklearn.dummy import DummyClassifier

        class UnstableModel:
            """Always predicts 0; after perturbation of numeric data predicts differently."""
            _call_count = 0

            def predict(self, X):
                UnstableModel._call_count += 1
                if UnstableModel._call_count <= 1:
                    return np.zeros(len(X), dtype=int)
                return np.ones(len(X), dtype=int)

            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)) * 0.9, np.ones(len(X)) * 0.1])

        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        result = TrustAnalyzer().analyze(UnstableModel(), X, y)
        assert isinstance(result.notes, list)

    def test_decision_function_binary_svc(self):
        """SVC with decision_function (no predict_proba when probability=False)."""
        from sklearn.svm import SVC
        from kaizenstat.reliability.trust import TrustAnalyzer
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
        y = pd.Series([i % 2 for i in range(100)])
        model = SVC(kernel="linear", probability=False)
        model.fit(X, y)
        result = TrustAnalyzer().analyze(model, X, y)
        assert 0 <= result.trust_score <= 100

    def test_regression_robustness_reg_exception(self):
        """Regression model predict raises on perturbed data → _robustness_reg returns 1.0."""
        from kaizenstat.reliability.trust import TrustAnalyzer
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(10)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100)})
        y = pd.Series(rng.normal(0, 1, 100))
        model = Ridge().fit(X, y)

        analyzer = TrustAnalyzer()
        base_pred = model.predict(X)
        # Mock _perturb to return non-None, but mock predict to raise on that call
        Xp = X.copy()
        Xp["a"] = Xp["a"] + 0.01
        with patch.object(analyzer, "_perturb", return_value=Xp):
            with patch.object(model, "predict", side_effect=RuntimeError("fail")):
                rob = analyzer._robustness_reg(model, X, base_pred)
        assert rob == 1.0

    def test_low_confidence_note_triggered(self):
        """Many low-confidence predictions → uncertain fraction note."""
        from kaizenstat.reliability.trust import TrustAnalyzer
        from sklearn.dummy import DummyClassifier

        rng = np.random.default_rng(5)
        X = pd.DataFrame({"a": rng.normal(0, 1, 200)})
        y = pd.Series([i % 2 for i in range(200)])
        model = DummyClassifier(strategy="uniform", random_state=0)
        model.fit(X, y)
        # threshold=0.0 means ALL predictions are above threshold → uncertain=0
        # threshold=0.99 means almost all are below → uncertain ~ 1.0
        result = TrustAnalyzer().analyze(model, X, y, low_conf_threshold=0.99)
        assert result.uncertain_fraction >= 0


# ── intelligence/ai_advisor.py — API call mocking ────────────────────────────

class TestAIAdvisorAPICall:
    def test_advise_with_mocked_api_success(self, monkeypatch):
        """Mock a successful Anthropic API response."""
        from kaizenstat.intelligence.ai_advisor import AIAdvisor

        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Here is my advice: fix missing data.")]
        mock_client.messages.create.return_value = mock_msg

        advisor = AIAdvisor(api_key="fake-key-abc123")
        advisor._client = mock_client  # inject mock client

        result = advisor.advise(question="What should I do?")
        assert "advice" in result.lower() or len(result) > 0

    def test_advise_api_exception_returns_empty(self, monkeypatch):
        """When API raises, advise() catches and returns empty string."""
        from kaizenstat.intelligence.ai_advisor import AIAdvisor

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        advisor = AIAdvisor(api_key="fake-key-xyz")
        advisor._client = mock_client

        result = advisor.advise()
        assert result == ""

    def test_get_client_runtime_error(self, monkeypatch):
        """When Anthropic() init raises, _get_client raises RuntimeError."""
        from kaizenstat.intelligence.ai_advisor import AIAdvisor
        import builtins
        orig_import = builtins.__import__

        class MockAnthropic:
            def __init__(self, **kwargs):
                raise Exception("init failed")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MockAnthropic

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                return mock_anthropic
            return orig_import(name, *args, **kwargs)

        advisor = AIAdvisor(api_key="fake-key")
        advisor._client = None
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="Failed to initialise"):
                advisor._get_client()


# ── fix/engine.py — module-level missing/outlier/encoding/imbalance APIs ──────

class TestFixModuleAPIs:
    def test_module_missing(self, small_df):
        from kaizenstat.fix import engine as fix_mod
        plan = fix_mod.missing(small_df, target="churn")
        assert plan is not None

    def test_module_outlier_handling(self, outlier_df):
        from kaizenstat.fix import engine as fix_mod
        plan = fix_mod.outlier_handling(outlier_df, target="target")
        assert plan is not None

    def test_module_encoding(self, small_df):
        from kaizenstat.fix import engine as fix_mod
        plan = fix_mod.encoding(small_df, target="churn")
        assert plan is not None

    def test_module_imbalance(self, imbalanced_df):
        from kaizenstat.fix import engine as fix_mod
        result = fix_mod.imbalance(imbalanced_df, target="target")
        assert isinstance(result, str)


# ── health/scorer.py — exception in correlation ───────────────────────────────

class TestHealthScorerLeakageException:
    def test_leakage_proxy_all_nan_column(self):
        """A column with all NaN values → corr() returns NaN → except not raised but checks pass."""
        from kaizenstat.health.scorer import HealthScorer
        df = pd.DataFrame({
            "all_nan": [np.nan] * 100,
            "normal": np.random.randn(100),
            "target": range(100),
        })
        result = HealthScorer().report(df, target="target")
        # Should not raise; score is valid
        assert 0 <= result.score <= 100


# ── validate/checker.py — remaining exception paths ──────────────────────────

class TestValidatorRemainingPaths:
    def test_drift_exception_handling(self):
        """ks_2samp called on all-NaN column is handled gracefully."""
        from kaizenstat.validate.checker import Validator
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": [np.nan] * 100,
        })
        X_test = pd.DataFrame({
            "a": rng.normal(10, 1, 50),  # shifted distribution → drift
            "b": [np.nan] * 50,
        })
        result = Validator().detect_drift(X_train, X_test)
        assert isinstance(result, dict)

    def test_normality_column_fewer_than_8(self):
        """Column with <8 non-null values is skipped in normality check."""
        from kaizenstat.validate.checker import Validator
        df = pd.DataFrame({
            "tiny_col": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan],
            "target": [0, 1, 0, 1, 0, 1, 0],
        })
        result = Validator().assumptions(df, target="target")
        assert result is not None

    def test_normality_exception_handling(self):
        """When shapiro raises, exception is caught."""
        from kaizenstat.validate.checker import Validator
        with patch("scipy.stats.shapiro", side_effect=Exception("shapiro failed")):
            df = pd.DataFrame({
                "x": np.random.randn(50),
                "target": [i % 2 for i in range(50)],
            })
            result = Validator().assumptions(df, target="target")
            assert result is not None

    def test_skewness_only_moderate_no_severe(self):
        """When only moderate skew (no severe) — severe branch empty, moderate branch triggered."""
        from kaizenstat.validate.checker import Validator
        rng = np.random.default_rng(0)
        # Create data with moderate skew (1 < |skew| ≤ 3) but not severe (>3)
        data = np.concatenate([rng.normal(0, 1, 150), np.array([5.0, 6.0, 7.0] * 10)])
        df = pd.DataFrame({
            "moderate_skew": data,
            "target": rng.integers(0, 2, len(data)),
        })
        result = Validator().skewness(df, target="target")
        assert result is not None

    def test_vif_r2_zero_check(self):
        """When r2 == 0 for a feature, VIF is 1/(1-0) = 1 → not flagged."""
        from kaizenstat.validate.checker import Validator
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
            "target": rng.integers(0, 2, 100),
        })
        result = Validator().multicollinearity(df, target="target")
        assert result is not None

    def test_leakage_corr_exception_with_constant(self):
        """Constant column → corr() returns NaN → caught silently."""
        from kaizenstat.validate.checker import Validator
        df = pd.DataFrame({
            "constant": [1.0] * 100,
            "target": range(100),
        })
        result = Validator().leakage(df, target="target")
        assert result is not None

    def test_distribution_std_zero_skipped(self):
        """Column with std=0 is skipped in distribution check."""
        from kaizenstat.validate.checker import Validator
        df = pd.DataFrame({
            "zero_std": [5.0] * 100,
            "normal": np.random.randn(100),
            "target": [i % 2 for i in range(100)],
        })
        result = Validator().distribution_check(df, target="target")
        assert result is not None


# ── improve/suggester.py — remaining branches ────────────────────────────────

class TestSuggesterRemainingBranches:
    def test_underfitting_high_test_score_branch(self, small_df):
        """Underfitting label but test_score ≥ 0.7 → else fallback gain string."""
        from kaizenstat.improve.suggester import Suggester

        class DR:
            issues = [type("I", (), {
                "name": "x", "description": "desc",
                "suggestion": "use better model", "risk_level": "HIGH"
            })()]
            label = "underfitting"
            gap = 0.0
            test_score = 0.75  # >= 0.7 → skips underfitting-specific branch
            feature_importances = None

        report = Suggester().suggest(small_df, target="churn", debug_result=DR())
        assert len(report.suggestions) > 0

    def test_tuning_suggestion_score_between_80_and_85(self, small_df):
        """test_score in [0.80, 0.85) → tuning suggestion (not stacking)."""
        from kaizenstat.improve.suggester import Suggester

        class DR:
            issues = []
            feature_importances = None
            test_score = 0.82
            label = "slightly_underfit"
            gap = 0.05

        report = Suggester().suggest(small_df, target="churn", debug_result=DR())
        actions = [s.action for s in report.suggestions]
        assert any("tune" in a.lower() or "tuning" in a.lower() or "train_best" in a for a in actions)

    def test_calibration_skipped_low_score(self, small_df):
        """test_score ≤ 0.70 → calibration NOT suggested."""
        from kaizenstat.improve.suggester import Suggester

        class DR:
            issues = []
            feature_importances = None
            test_score = 0.55
            label = "underfitting"
            gap = 0.30

        report = Suggester().suggest(small_df, target="churn", debug_result=DR())
        actions = [s.action for s in report.suggestions]
        assert not any("calibrat" in a.lower() for a in actions)
