"""Tests for CLI commands (kaizenstat/cli/main.py) via typer CliRunner."""
import os
import tempfile

import pandas as pd
import pytest
from typer.testing import CliRunner

from kaizenstat.cli.main import app


runner = CliRunner()


@pytest.fixture
def csv_path(tmp_path):
    """Write a small clean CSV and return its path."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "age":    rng.integers(20, 65, n),
        "income": rng.normal(50000, 15000, n).round(2),
        "churn":  rng.integers(0, 2, n),
    })
    p = str(tmp_path / "data.csv")
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def regression_csv(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, 100),
        "x2": rng.normal(0, 1, 100),
        "target": rng.normal(0, 1, 100),
    })
    p = str(tmp_path / "reg.csv")
    df.to_csv(p, index=False)
    return p


# ── health ────────────────────────────────────────────────────────────────────

class TestCLIHealth:
    def test_health_with_target(self, csv_path):
        result = runner.invoke(app, ["health", csv_path, "--target", "churn"])
        assert result.exit_code == 0

    def test_health_without_target(self, csv_path):
        result = runner.invoke(app, ["health", csv_path])
        assert result.exit_code == 0

    def test_health_missing_file(self, tmp_path):
        result = runner.invoke(app, ["health", str(tmp_path / "nope.csv")])
        assert result.exit_code != 0


# ── validate ──────────────────────────────────────────────────────────────────

class TestCLIValidate:
    def test_validate_with_target(self, csv_path):
        result = runner.invoke(app, ["validate", csv_path, "--target", "churn"])
        assert result.exit_code == 0

    def test_validate_missing_file(self, tmp_path):
        result = runner.invoke(app, ["validate", str(tmp_path / "nope.csv")])
        assert result.exit_code != 0


# ── fix ───────────────────────────────────────────────────────────────────────

class TestCLIFix:
    def test_fix_preview(self, csv_path):
        result = runner.invoke(app, ["fix", csv_path, "--target", "churn", "--preview"])
        assert result.exit_code == 0

    def test_fix_apply_and_save(self, csv_path, tmp_path):
        out = str(tmp_path / "fixed.csv")
        result = runner.invoke(app, ["fix", csv_path, "--target", "churn", "--output", out])
        assert result.exit_code == 0
        assert os.path.exists(out)

    def test_fix_apply_default_output(self, tmp_path):
        import numpy as np
        rng = np.random.default_rng(9)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 50),
            "y": [None if i % 5 == 0 else i for i in range(50)],
            "t": rng.integers(0, 2, 50),
        })
        p = str(tmp_path / "mydata.csv")
        df.to_csv(p, index=False)
        result = runner.invoke(app, ["fix", p])
        assert result.exit_code == 0

    def test_fix_missing_file(self, tmp_path):
        result = runner.invoke(app, ["fix", str(tmp_path / "no.csv")])
        assert result.exit_code != 0


# ── train ─────────────────────────────────────────────────────────────────────

class TestCLITrain:
    def test_train_basic(self, csv_path):
        result = runner.invoke(app, ["train", csv_path, "churn", "--cv", "3"])
        assert result.exit_code == 0

    def test_train_with_export(self, csv_path, tmp_path):
        out = str(tmp_path / "model.joblib")
        result = runner.invoke(app, ["train", csv_path, "churn", "--cv", "3", "--export", out])
        assert result.exit_code == 0
        assert os.path.exists(out)

    def test_train_missing_file(self, tmp_path):
        result = runner.invoke(app, ["train", str(tmp_path / "none.csv"), "target"])
        assert result.exit_code != 0


# ── debug ─────────────────────────────────────────────────────────────────────

class TestCLIDebug:
    def test_debug_basic(self, csv_path):
        result = runner.invoke(app, ["debug", csv_path, "churn"])
        assert result.exit_code == 0

    def test_debug_missing_file(self, tmp_path):
        result = runner.invoke(app, ["debug", str(tmp_path / "no.csv"), "target"])
        assert result.exit_code != 0


# ── improve ───────────────────────────────────────────────────────────────────

class TestCLIImprove:
    def test_improve_with_target(self, csv_path):
        result = runner.invoke(app, ["improve", csv_path, "--target", "churn"])
        assert result.exit_code == 0

    def test_improve_no_target(self, csv_path):
        result = runner.invoke(app, ["improve", csv_path])
        assert result.exit_code == 0

    def test_improve_missing_file(self, tmp_path):
        result = runner.invoke(app, ["improve", str(tmp_path / "no.csv")])
        assert result.exit_code != 0


# ── report ────────────────────────────────────────────────────────────────────

class TestCLIReport:
    def test_report_generates_html(self, csv_path, tmp_path):
        out = str(tmp_path / "report.html")
        result = runner.invoke(app, ["report", csv_path, "--target", "churn", "--output", out])
        assert result.exit_code == 0
        assert os.path.exists(out)

    def test_report_missing_file(self, tmp_path):
        result = runner.invoke(app, ["report", str(tmp_path / "no.csv")])
        assert result.exit_code != 0


# ── codegen ───────────────────────────────────────────────────────────────────

class TestCLICodegen:
    def test_codegen_creates_script(self, csv_path, tmp_path):
        out = str(tmp_path / "pipeline.py")
        result = runner.invoke(app, ["codegen", csv_path, "churn", "--output", out])
        assert result.exit_code == 0
        assert os.path.exists(out)

    def test_codegen_regression(self, regression_csv, tmp_path):
        out = str(tmp_path / "reg_pipeline.py")
        result = runner.invoke(app, ["codegen", regression_csv, "target", "--output", out])
        assert result.exit_code == 0

    def test_codegen_missing_file(self, tmp_path):
        result = runner.invoke(app, ["codegen", str(tmp_path / "no.csv"), "target"])
        assert result.exit_code != 0


# ── export ────────────────────────────────────────────────────────────────────

class TestCLIExport:
    def test_export_saves_model(self, csv_path, tmp_path):
        out = str(tmp_path / "model.joblib")
        result = runner.invoke(app, ["export", csv_path, "churn", "--output", out, "--cv", "3"])
        assert result.exit_code == 0
        assert os.path.exists(out)

    def test_export_missing_file(self, tmp_path):
        result = runner.invoke(app, ["export", str(tmp_path / "no.csv"), "target"])
        assert result.exit_code != 0


# ── auto ──────────────────────────────────────────────────────────────────────

class TestCLIAuto:
    def test_auto_full_pipeline(self, csv_path, tmp_path):
        out = str(tmp_path / "auto_report.html")
        result = runner.invoke(app, ["auto", csv_path, "churn", "--output", out])
        assert result.exit_code == 0

    def test_auto_missing_file(self, tmp_path):
        result = runner.invoke(app, ["auto", str(tmp_path / "no.csv"), "target"])
        assert result.exit_code != 0
