"""Tests for output/reporter.py — summary, HTML, export/load, codegen."""
import os
import tempfile

import pandas as pd
import pytest

from kaizenstat.output import reporter as out_mod
from kaizenstat.output.reporter import Reporter


# ── Minimal stub objects ─────────────────────────────────────────────────────

class _Penalty:
    def __init__(self, name, penalty, risk_level, reason):
        self.name = name
        self.penalty = penalty
        self.risk_level = risk_level
        self.reason = reason


class _HealthResult:
    score = 85
    grade = "B"
    risk_level = "LOW"
    penalties = [_Penalty("Missing Values", -5.0, "MEDIUM", "10% missing")]
    summary = "Good health."


class _ValidationIssue:
    check = "Normality"
    risk_level = "MEDIUM"
    issue = "Non-normal features"
    recommendation = "Apply log transform"


class _ValidationResult:
    passed = True
    issues = [_ValidationIssue()]


class _TrainResult:
    model_name = "RandomForest"
    test_score = 0.88
    task = "classification"


class _DebugResult:
    gap = 0.05
    diagnosis = "healthy"
    root_cause = "None"
    train_score = 0.92
    test_score = 0.87
    issues = []


class _Suggestion:
    priority = 1
    category = "Data Quality"
    action = "Fix missing values"
    expected_gain = "+5% accuracy"
    impact = "HIGH"


class _ImprovementReport:
    suggestions = [_Suggestion()]
    top_priority = _Suggestion()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": range(10),
        "income": [float(i) for i in range(10)],
        "target": [i % 2 for i in range(10)],
    })


# ── Reporter.summary ─────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_all_results(self):
        r = Reporter()
        r.summary(
            health_result=_HealthResult(),
            validation_result=_ValidationResult(),
            train_result=_TrainResult(),
            debug_result=_DebugResult(),
            improvement_report=_ImprovementReport(),
        )

    def test_summary_no_results(self):
        r = Reporter()
        r.summary()

    def test_summary_health_only(self):
        r = Reporter()
        r.summary(health_result=_HealthResult())

    def test_summary_validation_failed(self):
        vr = _ValidationResult()
        vr.passed = False
        Reporter().summary(validation_result=vr)

    def test_summary_debug_large_gap(self):
        dr = _DebugResult()
        dr.gap = 0.25
        dr.diagnosis = "overfitting"
        Reporter().summary(debug_result=dr)

    def test_summary_improvement_no_top(self):
        ir = _ImprovementReport()
        ir.top_priority = None
        Reporter().summary(improvement_report=ir)


# ── Reporter.html ─────────────────────────────────────────────────────────────

class TestHTMLReport:
    def test_html_full_results(self, tmp_path):
        path = str(tmp_path / "report.html")
        results = {
            "health": _HealthResult(),
            "validation": _ValidationResult(),
            "debug": _DebugResult(),
            "improvements": _ImprovementReport(),
        }
        out = Reporter().html(results, path=path)
        assert out == path
        with open(path) as f:
            content = f.read()
        assert "KaizenStat" in content

    def test_html_empty_results(self, tmp_path):
        path = str(tmp_path / "empty.html")
        out = Reporter().html({}, path=path)
        assert os.path.exists(out)

    def test_html_no_penalties(self, tmp_path):
        health = _HealthResult()
        health.penalties = []
        path = str(tmp_path / "nopen.html")
        Reporter().html({"health": health}, path=path)
        with open(path) as f:
            content = f.read()
        assert "No penalties" in content

    def test_html_no_validation_issues(self, tmp_path):
        vr = _ValidationResult()
        vr.issues = []
        path = str(tmp_path / "noissues.html")
        Reporter().html({"validation": vr}, path=path)
        assert os.path.exists(path)

    def test_html_debug_large_gap(self, tmp_path):
        dr = _DebugResult()
        dr.gap = 0.20
        path = str(tmp_path / "gap.html")
        Reporter().html({"debug": dr}, path=path)
        assert os.path.exists(path)

    def test_html_debug_small_gap(self, tmp_path):
        dr = _DebugResult()
        dr.gap = 0.03
        path = str(tmp_path / "smallgap.html")
        Reporter().html({"debug": dr}, path=path)
        assert os.path.exists(path)

    def test_html_with_improvements(self, tmp_path):
        path = str(tmp_path / "improve.html")
        Reporter().html({"improvements": _ImprovementReport()}, path=path)
        assert os.path.exists(path)

    def test_html_various_risk_levels(self, tmp_path):
        """Ensure all risk level color-map branches are exercised."""
        for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]:
            health = _HealthResult()
            health.risk_level = risk
            path = str(tmp_path / f"risk_{risk}.html")
            Reporter().html({"health": health}, path=path)
            assert os.path.exists(path)


# ── Module-level html/summary ─────────────────────────────────────────────────

class TestModuleLevelAPIs:
    def test_module_summary(self):
        out_mod.summary(health_result=_HealthResult())

    def test_module_html(self, tmp_path):
        path = str(tmp_path / "mod.html")
        out = out_mod.html({}, path=path)
        assert os.path.exists(out)


# ── Reporter.export_model / load_model ───────────────────────────────────────

class TestModelExportLoad:
    def test_export_and_load_joblib(self, tmp_path):
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier()
        path = str(tmp_path / "model.joblib")
        saved = Reporter().export_model(model, path=path)
        assert saved == path
        loaded = Reporter().load_model(path)
        assert loaded is not None

    def test_module_export_model(self, tmp_path):
        from sklearn.dummy import DummyClassifier
        path = str(tmp_path / "mod_model.joblib")
        out_mod.export_model(DummyClassifier(), path=path)
        assert os.path.exists(path)

    def test_module_load_model(self, tmp_path):
        from sklearn.dummy import DummyClassifier
        path = str(tmp_path / "load_model.joblib")
        out_mod.export_model(DummyClassifier(), path=path)
        m = out_mod.load_model(path)
        assert m is not None


# ── Reporter.codegen ──────────────────────────────────────────────────────────

class TestCodegen:
    def test_codegen_creates_file(self, sample_df, tmp_path):
        path = str(tmp_path / "pipeline.py")
        out = Reporter().codegen(sample_df, target="target", output_path=path)
        assert out == path
        with open(path) as f:
            content = f.read()
        assert "target" in content
        assert "KaizenStat" in content

    def test_codegen_regression_task(self, sample_df, tmp_path):
        path = str(tmp_path / "reg_pipeline.py")
        Reporter().codegen(sample_df, target="target", output_path=path, task="regression")
        with open(path) as f:
            content = f.read()
        assert "regression" in content

    def test_codegen_no_target_raises(self, sample_df, tmp_path):
        path = str(tmp_path / "pipeline.py")
        with pytest.raises(ValueError, match="target"):
            Reporter().codegen(sample_df, target=None, output_path=path)

    def test_module_codegen(self, sample_df, tmp_path):
        path = str(tmp_path / "mod_pipeline.py")
        out = out_mod.codegen(sample_df, target="target", output_path=path)
        assert os.path.exists(out)
