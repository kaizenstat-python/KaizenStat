"""Output engine — terminal reports, HTML export, model serialisation, code generation."""
from __future__ import annotations

import datetime
import html as _html
import os
from typing import Any, Dict, Optional

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class Reporter:
    """Generates reports and exports artefacts from the KaizenStat pipeline."""

    # ------------------------------------------------------------------ #
    # Terminal output                                                      #
    # ------------------------------------------------------------------ #

    def summary(
        self,
        health_result: Optional[Any] = None,
        validation_result: Optional[Any] = None,
        train_result: Optional[Any] = None,
        debug_result: Optional[Any] = None,
        improvement_report: Optional[Any] = None,
    ) -> None:
        """Print a concise pipeline summary to the terminal."""
        console.print(Panel.fit(
            "[bold cyan]KaizenStat Pipeline Summary[/bold cyan]",
            border_style="cyan",
        ))

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan", width=22)
        table.add_column("Status / Score", justify="right")
        table.add_column("Key Finding")

        if health_result is not None:
            grade_color = {"A": "green", "B": "green", "C": "yellow",
                           "D": "red", "F": "bold red"}.get(health_result.grade, "white")
            table.add_row(
                "Data Health",
                f"[{grade_color}]{health_result.score}/100 ({health_result.grade})[/{grade_color}]",
                health_result.summary,
            )

        if validation_result is not None:
            status = "[green]PASSED[/green]" if validation_result.passed else "[red]FAILED[/red]"
            issues_txt = f"{len(validation_result.issues)} issue(s)" if validation_result.issues else "All clear"
            table.add_row("Validation", status, issues_txt)

        if train_result is not None:
            table.add_row(
                "Model Training",
                f"[bold]{train_result.test_score:.4f}[/bold]",
                f"{train_result.model_name} · test {train_result.task}",
            )

        if debug_result is not None:
            diag_color = "green" if debug_result.diagnosis == "healthy" else "red"
            table.add_row(
                "Model Debug",
                f"gap=[{diag_color}]{debug_result.gap:+.4f}[/{diag_color}]",
                debug_result.diagnosis.upper(),
            )

        if improvement_report is not None:
            n = len(improvement_report.suggestions)
            top = improvement_report.top_priority
            top_txt = top.action[:60] if top else "No suggestions"
            table.add_row("Improvements", f"{n} suggestion(s)", top_txt)

        console.print(table)
        console.print()

    # ------------------------------------------------------------------ #
    # HTML report                                                          #
    # ------------------------------------------------------------------ #

    def html(
        self,
        results: Dict[str, Any],
        path: str = "kaizenstat_report.html",
        open_browser: bool = False,
    ) -> str:
        """Generate a self-contained HTML report."""
        html = self._build_html(results)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        console.print(f"[bold green]✓ HTML report saved → {path}[/bold green]")
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(path)}")
        return path

    # ------------------------------------------------------------------ #
    # Model export                                                         #
    # ------------------------------------------------------------------ #

    def export_model(self, model: Any, path: str = "model.joblib") -> str:
        """Save a trained pipeline to disk."""
        try:
            import joblib
            joblib.dump(model, path)
            console.print(f"[bold green]✓ Model saved → {path}[/bold green]")
            return path
        except ImportError:
            import pickle
            pkl_path = path.replace(".joblib", ".pkl")
            with open(pkl_path, "wb") as f:
                import pickle
                pickle.dump(model, f)
            console.print(f"[bold green]✓ Model saved (pickle) → {pkl_path}[/bold green]")
            return pkl_path

    def load_model(self, path: str) -> Any:
        """Load a previously exported pipeline."""
        try:
            import joblib
            model = joblib.load(path)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                model = pickle.load(f)
        console.print(f"[bold green]✓ Model loaded from {path}[/bold green]")
        return model

    # ------------------------------------------------------------------ #
    # Code generation                                                      #
    # ------------------------------------------------------------------ #

    def codegen(
        self,
        df: pd.DataFrame,
        target: Optional[str],
        output_path: str = "pipeline.py",
        task: str = "classification",
    ) -> str:
        """Generate a standalone Python script that reproduces the KaizenStat pipeline."""
        if not target:
            raise ValueError("A target column is required for code generation.")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        n_rows, n_cols = df.shape

        code = f'''#!/usr/bin/env python3
"""
KaizenStat Auto-Generated Pipeline
Generated: {timestamp}
Dataset:   {n_rows} rows × {n_cols} columns
Target:    {target}
Task:      {task}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("your_data.csv")
target = "{target}"

# ── 2. Health check (KaizenStat) ──────────────────────────────────────────────
from kaizenstat import health
health_result = health.report(df, target=target)
print(f"Health Score: {{health_result.score}}/100")

# ── 3. Fix data ────────────────────────────────────────────────────────────────
from kaizenstat import fix
fix_plan = fix.plan(df, target=target, safe=True)
df = fix_plan.apply(df)

# ── 4. Prepare features ────────────────────────────────────────────────────────
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
])

# ── 5. Train/test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 6. Build pipeline ──────────────────────────────────────────────────────────
model = {"RandomForestClassifier(n_estimators=100, random_state=42)" if task == "classification"
         else "RandomForestRegressor(n_estimators=100, random_state=42)"}
pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

# ── 7. Train ───────────────────────────────────────────────────────────────────
pipe.fit(X_train, y_train)
train_score = pipe.score(X_train, y_train)
test_score  = pipe.score(X_test, y_test)
print(f"Train Score: {{train_score:.4f}}")
print(f"Test Score:  {{test_score:.4f}}")

# ── 8. Debug ───────────────────────────────────────────────────────────────────
from kaizenstat import debug
debug.model_failure(pipe, X_train, X_test, y_train, y_test)

# ── 9. Export model ────────────────────────────────────────────────────────────
import joblib
joblib.dump(pipe, "model.joblib")
print("Model exported to model.joblib")
'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        console.print(f"[bold green]✓ Pipeline code generated → {output_path}[/bold green]")
        return output_path

    # ------------------------------------------------------------------ #
    # HTML builder                                                         #
    # ------------------------------------------------------------------ #

    def _build_html(self, results: Dict[str, Any]) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        health = results.get("health")
        validation = results.get("validation")
        debug = results.get("debug")
        improvements = results.get("improvements")

        def safe_score(val, default="—"):  # pragma: no cover
            return f"{val:.4f}" if val is not None else default

        health_score = health.score if health else "—"
        health_grade = health.grade if health else "—"
        health_risk = health.risk_level if health else "—"

        e = _html.escape  # shorthand for HTML-escaping user data
        penalty_rows = ""
        if health:
            for p in health.penalties:
                penalty_rows += (
                    f"<tr><td>{e(p.name)}</td><td class='num'>{p.penalty:+.1f}</td>"
                    f"<td><span class='badge badge-{e(p.risk_level.lower())}'>{e(p.risk_level)}</span></td>"
                    f"<td>{e(p.reason)}</td></tr>\n"
                )

        validation_rows = ""
        if validation:
            for iss in validation.issues:
                validation_rows += (
                    f"<tr><td>{e(iss.check)}</td>"
                    f"<td><span class='badge badge-{e(iss.risk_level.lower())}'>{e(iss.risk_level)}</span></td>"
                    f"<td>{e(iss.issue)}</td><td>{e(iss.recommendation)}</td></tr>\n"
                )

        debug_section = ""
        if debug:
            gap_color = "red" if abs(debug.gap) > 0.10 else "#f0ad4e" if abs(debug.gap) > 0.05 else "green"
            debug_section = f"""
            <div class="card">
              <h2>Model Debug</h2>
              <div class="score-box">
                <div class="score-item"><span class="label">Train Score</span><span class="value">{debug.train_score:.4f}</span></div>
                <div class="score-item"><span class="label">Test Score</span><span class="value">{debug.test_score:.4f}</span></div>
                <div class="score-item"><span class="label">Gap</span><span class="value" style="color:{gap_color}">{debug.gap:+.4f}</span></div>
                <div class="score-item"><span class="label">Diagnosis</span><span class="value">{e(debug.diagnosis.upper())}</span></div>
              </div>
              <p><strong>Root Cause:</strong> {e(debug.root_cause)}</p>
            </div>"""

        improve_rows = ""
        if improvements:
            for s in improvements.suggestions[:10]:
                improve_rows += (
                    f"<tr><td>{s.priority}</td><td>{e(s.category)}</td>"
                    f"<td><span class='badge badge-{e(s.impact.lower())}'>{e(s.impact)}</span></td>"
                    f"<td>{e(s.action)}</td><td>{e(s.expected_gain)}</td></tr>\n"
                )

        risk_color_map = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545", "CRITICAL": "#6f0000"}
        risk_color = risk_color_map.get(health_risk, "#888")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KaizenStat Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f8f9fa; color: #212529; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #0d6efd, #0dcaf0);
             color: #fff; padding: 2rem; text-align: center; }}
  .header h1 {{ font-size: 2rem; margin-bottom: .25rem; }}
  .header p  {{ opacity: .8; font-size: .9rem; }}
  .container {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
           padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2 {{ font-size: 1.2rem; color: #0d6efd; border-bottom: 2px solid #e9ecef;
              padding-bottom: .5rem; margin-bottom: 1rem; }}
  .score-box {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  .score-item {{ background: #f1f3f5; border-radius: 8px; padding: .75rem 1.25rem; text-align: center; }}
  .score-item .label {{ display: block; font-size: .75rem; color: #6c757d; text-transform: uppercase; }}
  .score-item .value {{ display: block; font-size: 1.8rem; font-weight: 700; }}
  .health-score {{ font-size: 3rem; font-weight: 900; color: #0d6efd; }}
  .risk-badge {{ display: inline-block; padding: .25rem .75rem; border-radius: 20px;
                 color: #fff; font-weight: 600; background: {risk_color}; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th {{ background: #f1f3f5; padding: .6rem; text-align: left; border-bottom: 2px solid #dee2e6; }}
  td {{ padding: .5rem .6rem; border-bottom: 1px solid #f1f3f5; }}
  tr:hover {{ background: #f8f9fa; }}
  .num {{ text-align: right; font-family: monospace; }}
  .badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 4px;
            font-size: .75rem; font-weight: 700; color: #fff; }}
  .badge-low      {{ background: #28a745; }}
  .badge-medium   {{ background: #ffc107; color: #212529; }}
  .badge-high     {{ background: #dc3545; }}
  .badge-critical {{ background: #6f0000; }}
  .footer {{ text-align: center; color: #6c757d; font-size: .8rem; padding: 2rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>KaizenStat Report</h1>
  <p>Generated {ts}</p>
</div>
<div class="container">

  <div class="card">
    <h2>Data Health Score</h2>
    <div class="score-box">
      <div class="score-item">
        <span class="label">Score</span>
        <span class="value health-score">{health_score}</span>
      </div>
      <div class="score-item">
        <span class="label">Grade</span>
        <span class="value">{health_grade}</span>
      </div>
      <div class="score-item">
        <span class="label">Risk Level</span>
        <span class="value"><span class="risk-badge">{health_risk}</span></span>
      </div>
    </div>
    {"<table><thead><tr><th>Issue</th><th>Penalty</th><th>Risk</th><th>Reason</th></tr></thead><tbody>" + penalty_rows + "</tbody></table>" if penalty_rows else "<p style='color:#28a745'>✓ No penalties — dataset in excellent health.</p>"}
  </div>

  {"<div class='card'><h2>Validation</h2><table><thead><tr><th>Check</th><th>Risk</th><th>Issue</th><th>Recommendation</th></tr></thead><tbody>" + validation_rows + "</tbody></table></div>" if validation_rows else ""}

  {debug_section}

  {"<div class='card'><h2>Improvement Plan</h2><table><thead><tr><th>#</th><th>Category</th><th>Impact</th><th>Action</th><th>Expected Gain</th></tr></thead><tbody>" + improve_rows + "</tbody></table></div>" if improve_rows else ""}

</div>
<div class="footer">
  Generated by <strong>KaizenStat</strong> · The Data Health &amp; ML Debugging Framework
</div>
</body>
</html>"""


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_reporter = Reporter()


def summary(**kwargs) -> None:
    _reporter.summary(**kwargs)


def html(results: Dict[str, Any], path: str = "kaizenstat_report.html",
         open_browser: bool = False) -> str:
    return _reporter.html(results, path=path, open_browser=open_browser)


def export_model(model: Any, path: str = "model.joblib") -> str:
    return _reporter.export_model(model, path=path)


def load_model(path: str) -> Any:
    return _reporter.load_model(path)


def codegen(df: pd.DataFrame, target: str, output_path: str = "pipeline.py",
            task: str = "classification") -> str:
    return _reporter.codegen(df, target, output_path=output_path, task=task)
