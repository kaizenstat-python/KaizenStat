"""KaizenStat CLI — kz <command> [options]"""
from __future__ import annotations

import os

import typer
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()
app = typer.Typer(
    name="kz",
    help="KaizenStat — Data Health & ML Model Debugging Framework",
    add_completion=False,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _load(file: str) -> pd.DataFrame:
    if not os.path.exists(file):
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)
    return pd.read_csv(file)


def _header():
    console.print(Panel.fit(
        "[bold cyan]KaizenStat[/bold cyan]  ·  Data Health & ML Debugging",
        border_style="cyan",
    ))


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def health(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name"),
):
    """Compute and display the Data Health Score."""
    _header()
    df = _load(file)
    from kaizenstat.health.scorer import HealthScorer
    scorer = HealthScorer()
    scorer.report(df, target=target)


@app.command()
def validate(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name"),
):
    """Run statistical assumption and leakage validation checks."""
    _header()
    df = _load(file)
    from kaizenstat.validate.checker import Validator
    result = Validator().assumptions(df, target=target)
    result.display()


@app.command()
def fix(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name"),
    preview: bool = typer.Option(False, "--preview", "-p", is_flag=True, help="Preview only — do not apply"),
    output: str = typer.Option(None, "--output", "-o", help="Save fixed CSV to this path"),
):
    """Preview or apply safe data corrections."""
    _header()
    df = _load(file)
    from kaizenstat.fix.engine import FixEngine
    engine = FixEngine()
    if preview:
        engine.plan(df, target=target)
        return

    fixed = engine.apply(df, target=target)
    out = output or file.replace(".csv", "_fixed.csv")
    fixed.to_csv(out, index=False)
    console.print(f"[bold green]✓ Fixed dataset saved → {out}[/bold green]")


@app.command()
def train(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Target column name"),
    cv: int = typer.Option(5, "--cv", help="Cross-validation folds"),
    tune: bool = typer.Option(False, "--tune", is_flag=True, help="Auto-tune hyperparameters with RandomizedSearchCV"),
    n_iter: int = typer.Option(20, "--n-iter", help="Number of hyperparameter combinations to try (used with --tune)"),
    export: str = typer.Option(None, "--export", "-e", help="Export best model to this path"),
):
    """Benchmark models and train the best one. Use --tune to auto-tune hyperparameters."""
    _header()
    df = _load(file)
    from kaizenstat.model.trainer import ModelTrainer
    trainer = ModelTrainer()
    result = trainer.train_best(df, target, test_size=0.2, cv=cv, tune=tune, n_iter=n_iter)
    if export:
        from kaizenstat.output.reporter import Reporter
        Reporter().export_model(result.pipeline, path=export)


@app.command()
def debug(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Target column name"),
):
    """Run the full model debug analysis."""
    _header()
    df = _load(file)
    from kaizenstat.doctor.data_doctor import DataDoctor
    doctor = DataDoctor()
    doctor.fit(df, target=target)
    doctor.train()
    doctor.debug_model()


@app.command()
def improve(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name"),
):
    """Generate prioritised data and model improvement suggestions."""
    _header()
    df = _load(file)
    from kaizenstat.improve.suggester import Suggester
    from kaizenstat.health.scorer import HealthScorer
    from kaizenstat.validate.checker import Validator

    hr = HealthScorer().breakdown(df, target)
    vr = Validator().assumptions(df, target)
    Suggester().suggest(df, target=target, health_result=hr, validation_result=vr)


@app.command()
def report(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name"),
    output: str = typer.Option("kaizenstat_report.html", "--output", "-o", help="HTML output path"),
    open_browser: bool = typer.Option(False, "--open", is_flag=True, help="Open in browser"),
):
    """Generate a full HTML pipeline report."""
    _header()
    df = _load(file)
    from kaizenstat.health.scorer import HealthScorer
    from kaizenstat.validate.checker import Validator
    from kaizenstat.improve.suggester import Suggester
    from kaizenstat.output.reporter import Reporter

    hr = HealthScorer().breakdown(df, target)
    vr = Validator().assumptions(df, target)
    ir = Suggester().suggest(df, target=target, health_result=hr, validation_result=vr)

    results = {"health": hr, "validation": vr, "improvements": ir}
    Reporter().html(results, path=output, open_browser=open_browser)


@app.command()
def auto(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Target column name"),
    output: str = typer.Option("kaizenstat_report.html", "--output", "-o", help="HTML output path"),
):
    """Run the full KaizenStat pipeline in one command."""
    _header()
    df = _load(file)
    from kaizenstat.doctor.data_doctor import DataDoctor

    doctor = DataDoctor()
    doctor.fit(df, target=target)
    doctor.health()
    doctor.validate()
    doctor.fix(safe=True)
    doctor.train()
    doctor.debug_model()
    doctor.improve()
    doctor.report(output_path=output)


@app.command()
def codegen(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Target column name"),
    output: str = typer.Option("pipeline.py", "--output", "-o", help="Output .py file path"),
):
    """Generate a standalone Python training script from your dataset."""
    _header()
    df = _load(file)
    from kaizenstat.output.reporter import Reporter
    from kaizenstat.utils.helpers import detect_task_type
    task = detect_task_type(df[target].dropna())
    path = Reporter().codegen(df, target=target, output_path=output, task=task)
    console.print(f"[bold green]✓ Script saved → {path}[/bold green]")


@app.command()
def export(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Target column name"),
    output: str = typer.Option("model.joblib", "--output", "-o", help="Output .joblib file path"),
    cv: int = typer.Option(5, "--cv", help="Cross-validation folds"),
):
    """Train the best model and save it to a .joblib file."""
    _header()
    df = _load(file)
    from kaizenstat.model.trainer import ModelTrainer
    from kaizenstat.output.reporter import Reporter
    result = ModelTrainer().train_best(df, target, test_size=0.2, cv=cv)
    Reporter().export_model(result.pipeline, path=output)


def main():
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
