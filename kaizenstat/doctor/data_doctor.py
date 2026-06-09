"""DataDoctor — the primary sklearn-style orchestrator for the KaizenStat pipeline."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel

try:
    import polars  # noqa: F401
except ImportError:
    Console().print("[dim]Installing polars for fast data loading...[/dim]")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polars", "-q"])

from kaizenstat.debug.debugger import ModelDebugger
from kaizenstat.debug.text_debugger import TextModelDebugger
from kaizenstat.fix.engine import FixEngine
from kaizenstat.health.scorer import HealthResult, HealthScorer
from kaizenstat.health.text_scorer import TextHealthScorer
from kaizenstat.improve.suggester import ImprovementReport, Suggester
from kaizenstat.improve.text_suggester import TextSuggester
from kaizenstat.model.trainer import ModelTrainer, TrainResult
from kaizenstat.model.text_trainer import TextModelTrainer
from kaizenstat.output.reporter import Reporter
from kaizenstat.reliability.trust import TrustAnalyzer, TrustReport
from kaizenstat.utils.helpers import (
    detect_task_type,
    dominant_text_column,
    validate_dataframe,
)
from kaizenstat.validate.checker import ValidationReport, Validator
from kaizenstat.validate.text_checker import TextValidator

console = Console()


@dataclass
class ComparisonResult:
    """Before-vs-after result from auto_improve()."""
    before: TrainResult
    after: TrainResult

    @property
    def score_delta(self) -> float:
        return round(self.after.test_score - self.before.test_score, 4)

    def display(self) -> None:
        delta = self.score_delta
        color = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        console.print(Panel.fit(
            f"[bold]Before[/bold]  {self.before.model_name:20s}  Test: {self.before.test_score:.4f}\n"
            f"[bold]After[/bold]   {self.after.model_name:20s}  Test: {self.after.test_score:.4f}\n"
            f"[bold]Delta[/bold]   [{color}]{delta:+.4f}[/{color}]"
            + (f"  ({'improved' if delta > 0 else 'no gain'})" if delta != 0 else "  (no change)"),
            title="[bold]KaizenStat · Before vs After[/bold]",
            border_style=color,
        ))


class DataDoctor:
    """
    KaizenStat's primary interface — mirrors the sklearn API style.

    Typical workflow::

        doctor = DataDoctor()
        doctor.fit(df, target="survived")
        doctor.health()
        doctor.validate()
        doctor.fix(safe=True)
        doctor.train()
        doctor.debug_model()
        doctor.improve()
        doctor.report()
    """

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._target: Optional[str] = None
        self._fixed_df: Optional[pd.DataFrame] = None

        # Mode: "tabular" or "text" — auto-detected on fit()
        self._mode: str = "tabular"
        self._text_col: Optional[str] = None

        # Pipeline results (populated lazily)
        self._health_result: Optional[HealthResult] = None
        self._validation_result: Optional[ValidationReport] = None
        self._train_result: Optional[TrainResult] = None
        self._debug_result = None
        self._improvement_report: Optional[ImprovementReport] = None
        self._last_split = None       # cached (X_tr, X_te, y_tr, y_te, task) from debug_model()

        # Plugin registries
        self._custom_models: Dict[str, Any] = {}
        self._custom_checks: List[Callable] = []

        # Tabular module instances
        self._scorer = HealthScorer()
        self._validator = Validator()
        self._fix_engine = FixEngine()
        self._trainer = ModelTrainer()
        self._debugger = ModelDebugger()
        self._suggester = Suggester()
        self._reporter = Reporter()

        # Text (NLP) module instances
        self._text_scorer = TextHealthScorer()
        self._text_validator = TextValidator()
        self._text_trainer = TextModelTrainer()
        self._text_debugger = TextModelDebugger()
        self._text_suggester = TextSuggester()

        # Reliability / trust layer (mode-agnostic)
        self._trust = TrustAnalyzer()

    # ------------------------------------------------------------------ #
    # 0. load                                                              #
    # ------------------------------------------------------------------ #

    def load(self, path: str, **kwargs) -> "DataDoctor":
        """
        Load any file or URL in one command — no pandas boilerplate needed.

        Supported: .csv, .tsv, .xlsx, .xls, .parquet, .json, .jsonl, .feather
        Also accepts HTTP/HTTPS URLs.

        Uses Polars (Rust) when available for fast loading, falls back to pandas.

        Example::

            doctor = DataDoctor()
            doctor.load("titanic.csv")
            doctor.load("https://example.com/data.parquet")
            doctor.fit(target="Survived")
        """
        import time
        from rich.table import Table
        from rich.text import Text

        p = path.strip()
        is_url = p.startswith("http://") or p.startswith("https://")
        raw_ext = p.split("?")[0].split("#")[0].rsplit(".", 1)
        ext = ("." + raw_ext[-1].lower()) if len(raw_ext) == 2 else ""

        # Try Polars (Rust) first for speed; fall back to pandas
        engine = "pandas"
        t0 = time.perf_counter()
        try:
            import polars as pl
            if ext in (".csv", ".tsv", ""):
                sep = kwargs.pop("sep", "\t" if ext == ".tsv" else ",")
                df = pl.read_csv(p, separator=sep, **{k: v for k, v in kwargs.items() if k in ("has_header", "null_values", "skip_rows", "n_rows")}).to_pandas()
            elif ext == ".parquet":
                df = pl.read_parquet(p).to_pandas()
            elif ext == ".json":
                df = pl.read_json(p).to_pandas()
            elif ext == ".jsonl":
                df = pl.read_ndjson(p).to_pandas()
            elif ext in (".feather", ".ipc", ".arrow"):
                df = pl.read_ipc(p).to_pandas()
            else:
                raise ImportError("unsupported by polars")
            engine = "polars"
        except Exception:
            # Polars unavailable or unsupported format — use pandas
            try:
                if ext in (".csv", ".tsv", ""):
                    sep = kwargs.pop("sep", "\t" if ext == ".tsv" else ",")
                    df = pd.read_csv(p, sep=sep, **kwargs)
                elif ext in (".xlsx", ".xls"):
                    df = pd.read_excel(p, **kwargs)
                elif ext == ".parquet":
                    df = pd.read_parquet(p, **kwargs)
                elif ext == ".json":
                    df = pd.read_json(p, **kwargs)
                elif ext == ".jsonl":
                    df = pd.read_json(p, lines=True, **kwargs)
                elif ext in (".feather", ".ipc", ".arrow"):
                    df = pd.read_feather(p, **kwargs)
                else:
                    df = pd.read_csv(p, **kwargs)
            except Exception as exc:
                raise ValueError(
                    f"Could not load '{path}'.\n"
                    f"Supported: csv, tsv, xlsx, xls, parquet, json, jsonl, feather\n"
                    f"Error: {exc}"
                ) from exc

        elapsed = time.perf_counter() - t0

        # Normalize pandas 2.x extension dtypes
        df = df.convert_dtypes(convert_string=False)

        self._df = df
        self._fixed_df = None
        self._target = None
        self._health_result = None
        self._validation_result = None
        self._train_result = None
        self._debug_result = None
        self._improvement_report = None

        rows, cols_count = df.shape
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        source_label = "URL" if is_url else "file"
        engine_label = f"[green]Polars ⚡[/green]" if engine == "polars" else "[dim]pandas[/dim]"

        # ── Summary panel ──────────────────────────────────────────────────
        summary = (
            f"[bold cyan]{source_label}:[/bold cyan] {p}\n"
            f"[bold]{rows:,}[/bold] rows  ×  [bold]{cols_count}[/bold] columns  "
            f"│  {mem_mb:.2f} MB  │  {elapsed:.2f}s  │  engine: {engine_label}\n"
        )

        # ── dtype breakdown ────────────────────────────────────────────────
        numeric_cols  = df.select_dtypes(include="number").columns.tolist()
        object_cols   = df.select_dtypes(include=["object", "category"]).columns.tolist()
        other_cols    = [c for c in df.columns if c not in numeric_cols and c not in object_cols]

        summary += (
            f"\n[bold]Numeric[/bold] ({len(numeric_cols)}): {numeric_cols[:8]}"
            + (" …" if len(numeric_cols) > 8 else "") + "\n"
            f"[bold]Categorical[/bold] ({len(object_cols)}): {object_cols[:8]}"
            + (" …" if len(object_cols) > 8 else "")
        )
        if other_cols:
            summary += f"\n[bold]Other[/bold] ({len(other_cols)}): {other_cols[:8]}"

        # ── missing values ─────────────────────────────────────────────────
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols):
            summary += f"\n\n[yellow]Missing values in {len(missing_cols)} column(s):[/yellow] "
            summary += "  ".join(f"{c}={v}" for c, v in missing_cols.items())
        else:
            summary += "\n\n[green]✓ No missing values[/green]"

        summary += f"\n\n[dim]Next → doctor.fit(target='column_name')[/dim]"

        console.print(Panel(summary, title="[bold]📂 KaizenStat · Load[/bold]", border_style="cyan", expand=False))

        # ── Preview table (first 5 rows) ───────────────────────────────────
        preview = Table(show_header=True, header_style="bold magenta", show_lines=True, expand=False)
        display_cols = list(df.columns[:10])
        if len(df.columns) > 10:
            console.print(f"[dim]Showing first 10 of {cols_count} columns[/dim]")

        for col in display_cols:
            preview.add_column(str(col), overflow="ellipsis", max_width=14, no_wrap=True)

        def _fmt(val):
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        for _, row in df.head(5).iterrows():
            preview.add_row(*[_fmt(row[c]) for c in display_cols])

        console.print(preview)

        return self

    # ------------------------------------------------------------------ #
    # 1. fit                                                               #
    # ------------------------------------------------------------------ #

    def fit(self, df: Optional[pd.DataFrame] = None, target: Optional[str] = None, keep_suspicious: bool = False) -> "DataDoctor":
        """
        Register a dataset with the doctor.

        Args:
            df:     Input pandas DataFrame. Omit if you already called load().
            target: Name of the target column (required for supervised tasks).
        """
        if df is None:
            if self._df is None:
                raise RuntimeError(
                    "No data loaded. Either pass a DataFrame: doctor.fit(df, target='...') "
                    "or load a file first: doctor.load('file.csv') then doctor.fit(target='...')."
                )
            # Already loaded via load() — just update target
        else:
            validate_dataframe(df, target)
            self._df = df.copy()

        # Normalize pandas 2.x extension dtypes (StringDtype etc.) — always, regardless of load path
        self._df = self._df.convert_dtypes(convert_string=False)

        if target is not None:
            # Validate target exists in whichever df we have
            if target not in self._df.columns:
                raise ValueError(
                    f"Target column '{target}' not found. "
                    f"Available columns: {list(self._df.columns)}"
                )
        self._target = target
        self._fixed_df = None

        # Reset cached results when re-fit
        self._health_result = None
        self._validation_result = None
        self._train_result = None
        self._debug_result = None
        self._improvement_report = None

        # ---- Auto-detect and warn about ID / high-cardinality text columns ----
        _excl = [target] if target else []
        _suspicious = _detect_suspicious_columns(self._df, exclude=_excl)
        if _suspicious:
            _names = ", ".join(f"'{c}'" for c in _suspicious)
            if keep_suspicious:
                console.print(
                    f"[dim]ℹ  Keeping potentially problematic columns (keep_suspicious=True): {_names}[/dim]"
                )
            else:
                console.print(
                    f"[yellow bold]⚠  KaizenStat detected columns that look like IDs or free-text "
                    f"and may break the ML pipeline:[/yellow bold]\n"
                    f"   {_names}\n"
                    f"[dim]   Auto-dropped. To keep them, pass keep_suspicious=True to fit().[/dim]"
                )
                self._df = self._df.drop(columns=_suspicious)

        # ---- Automatic mode detection (tabular vs text) ----
        self._text_col = dominant_text_column(self._df, exclude=[target] if target else None)
        self._mode = "text" if self._text_col is not None else "tabular"

        rows, cols = self._df.shape
        task_str = ""
        if target:
            task = detect_task_type(self._df[target].dropna())
            task_str = f"  │  Task: {task}"
        mode_str = (f"  │  Mode: [bold]{self._mode.upper()}[/bold]"
                    + (f" ('{self._text_col}')" if self._mode == "text" else ""))

        console.print(Panel.fit(
            f"[bold cyan]Dataset registered[/bold cyan]  "
            f"│  {rows:,} rows × {cols} columns{task_str}{mode_str}",
            title="[bold]DataDoctor.fit[/bold]",
            border_style="cyan",
        ))
        return self

    def mode(self) -> str:
        """Return the detected dataset mode: 'text' or 'tabular'."""
        self._require_fit()
        return self._mode

    # ------------------------------------------------------------------ #
    # 2. health                                                            #
    # ------------------------------------------------------------------ #

    def health(self) -> HealthResult:
        """Compute and display the Data Health Score (0–100). Routes to text scorer in text mode."""
        self._require_fit()
        if self._mode == "text":
            self._health_result = self._text_scorer.report(
                self._active_df, self._target, text_col=self._text_col
            )
        else:
            self._health_result = self._scorer.report(self._active_df, self._target)
        return self._health_result

    # ------------------------------------------------------------------ #
    # 3. validate                                                          #
    # ------------------------------------------------------------------ #

    def validate(self) -> ValidationReport:
        """Run statistical assumption and leakage checks (plus any registered custom checks).

        Routes to text validation (token skew, stopwords, rare-token explosion,
        text label leakage) when in text mode.
        """
        self._require_fit()
        if self._mode == "text":
            self._validation_result = self._text_validator.assumptions(
                self._active_df, self._target, text_col=self._text_col
            )
        else:
            self._validation_result = self._validator.assumptions(self._active_df, self._target)
        for label, fn in self._custom_checks:
            try:
                issues = fn(self._active_df, self._target)
                if issues:
                    console.print(f"[yellow]Custom check '[bold]{label}[/bold]':[/yellow]")
                    for iss in issues:
                        console.print(f"  • {iss}")
            except Exception as exc:
                console.print(f"[red]Custom check '{label}' raised: {exc}[/red]")
        self._validation_result.display()
        return self._validation_result

    # ------------------------------------------------------------------ #
    # 4. fix                                                               #
    # ------------------------------------------------------------------ #

    def fix(self, safe: bool = True, preview_only: bool = False) -> pd.DataFrame:
        """
        Show (and optionally apply) safe data corrections.

        Args:
            safe:         If True, only apply LOW-risk fixes.
            preview_only: If True, show the plan but do not apply it.

        Returns:
            Fixed DataFrame (or original if preview_only=True).
        """
        self._require_fit()
        plan = self._fix_engine.plan(self._active_df, self._target, safe=safe)
        if preview_only:
            return self._active_df
        self._fixed_df = plan.apply(self._active_df)
        return self._fixed_df

    # ------------------------------------------------------------------ #
    # 5. train                                                             #
    # ------------------------------------------------------------------ #

    def train(self, cv: int = 5, test_size: float = 0.2, tune: bool = False, n_iter: int = 20) -> TrainResult:
        """
        Benchmark models and train the best one.

        Args:
            cv:        Number of cross-validation folds.
            test_size: Fraction held out for the final test set.
            tune:      If True, run RandomizedSearchCV on the best model.
            n_iter:    Number of random hyperparameter combinations to try when tune=True.
        """
        self._require_fit()
        if not self._target:
            raise ValueError("A target column is required for training. Call fit(df, target=...).")
        if self._mode == "text":
            self._train_result = self._text_trainer.train_best(
                self._active_df, self._target, text_col=self._text_col,
                test_size=test_size, cv=cv, tune=tune,
                n_iter=min(n_iter, 15),
            )
        else:
            self._train_result = self._trainer.train_best(
                self._active_df, self._target,
                test_size=test_size, cv=cv, tune=tune, n_iter=n_iter,
                extra_models=self._custom_models or None,
            )
        return self._train_result

    # ------------------------------------------------------------------ #
    # 6. debug_model                                                       #
    # ------------------------------------------------------------------ #

    def debug_model(self, test_size: float = 0.2) -> Any:
        """
        Diagnose why the model is failing.

        Requires train() to have been called first, or will run it automatically.
        """
        self._require_fit()
        if not self._target:
            raise ValueError("A target column is required for model debugging.")
        if self._train_result is None:
            console.print("[dim]Running train() first...[/dim]")
            self.train(test_size=test_size)

        from sklearn.model_selection import train_test_split

        df = self._active_df.copy()
        # Drop rows with missing target to avoid stratify / scoring errors
        df = df.loc[df[self._target].notna()].copy()

        y = df[self._target]
        task = detect_task_type(y)

        # In text mode X is the single text column (Series of strings); otherwise the feature frame
        X = df[self._text_col].fillna("").astype(str) if self._mode == "text" \
            else df.drop(columns=[self._target])

        # Apply the same LabelEncoding the trainer used so pipeline scores are correct
        le = self._train_result.label_encoder
        if le is not None:
            y = pd.Series(le.transform(y), index=y.index, name=self._target)

        try:
            stratify = y if task == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify, random_state=42
            )
        except ValueError:
            # Fallback when a class has too few samples to stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Stash the split so trust_score() can reuse the exact same held-out test set
        self._last_split = (X_train, X_test, y_train, y_test, task)

        pipe = self._train_result.pipeline
        debugger = self._text_debugger if self._mode == "text" else self._debugger
        self._debug_result = debugger.model_failure(pipe, X_train, X_test, y_train, y_test)
        return self._debug_result

    # ------------------------------------------------------------------ #
    # 7. improve                                                           #
    # ------------------------------------------------------------------ #

    def improve(self) -> ImprovementReport:
        """Generate prioritised improvement suggestions from all pipeline stages.

        Routes to the NLP suggester (n-grams, embeddings, balancing, augmentation)
        in text mode.
        """
        self._require_fit()
        if self._mode == "text":
            self._improvement_report = self._text_suggester.suggest(
                self._active_df,
                target=self._target,
                text_col=self._text_col,
                health_result=self._health_result,
                debug_result=self._debug_result,
                validation_result=self._validation_result,
            )
        else:
            self._improvement_report = self._suggester.suggest(
                self._active_df,
                target=self._target,
                health_result=self._health_result,
                debug_result=self._debug_result,
                validation_result=self._validation_result,
            )
        return self._improvement_report

    # ------------------------------------------------------------------ #
    # 8. report                                                            #
    # ------------------------------------------------------------------ #

    def report(
        self,
        output_path: str = "kaizenstat_report.html",
        open_browser: bool = False,
    ) -> str:
        """
        Print a terminal summary and export an HTML report.

        Args:
            output_path:  File path for the HTML output.
            open_browser: Automatically open the report in a browser.

        Returns:
            Path to the generated HTML file.
        """
        self._require_fit()
        self._reporter.summary(
            health_result=self._health_result,
            validation_result=self._validation_result,
            train_result=self._train_result,
            debug_result=self._debug_result,
            improvement_report=self._improvement_report,
        )
        results = {
            "health": self._health_result,
            "validation": self._validation_result,
            "train": self._train_result,
            "debug": self._debug_result,
            "improvements": self._improvement_report,
        }
        return self._reporter.html(results, path=output_path, open_browser=open_browser)

    # ------------------------------------------------------------------ #
    # 9. auto_improve                                                      #
    # ------------------------------------------------------------------ #

    def train_auto(
        self,
        cv: int = 3,
        test_size: float = 0.2,
        tune: bool = False,
        n_iter: int = 10,
        ensemble: bool = True,
    ) -> TrainResult:
        """
        Full AutoML pipeline: profile data → smart model selection →
        optional tuning → soft-voting ensemble.

        Args:
            ensemble: If True, build a soft-voting ensemble of the top 3 models.
        """
        self._require_fit()
        if not self._target:
            raise ValueError("A target column is required. Call fit(df, target=...).")
        self._train_result = self._trainer.train_auto(
            self._active_df, self._target,
            test_size=test_size, cv=cv,
            tune=tune, n_iter=n_iter, ensemble=ensemble,
            extra_models=self._custom_models or None,
        )
        return self._train_result

    def detect_drift(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> dict:
        """
        Detect distribution drift between two datasets using the KS test.

        Returns {column: p_value} for features that show significant drift (p < 0.05).
        """
        return self._validator.detect_drift(X_train, X_test)

    def dataset_difficulty(self) -> float:
        """
        Estimate dataset difficulty (0 = trivially easy, 1 = near-impossible).

        Requires fit() to have been called.
        """
        self._require_fit()
        if not self._target:
            raise ValueError("A target column is required.")
        df = self._active_df.copy()
        df = df.loc[df[self._target].notna()]
        X  = df.drop(columns=[self._target])
        y  = df[self._target]
        return self._debugger.dataset_difficulty(X, y)

    def feature_impact(self, top_n: int = 15) -> dict:
        """
        Counterfactual feature impact: score drop when each feature is removed.

        Requires train() or train_auto() to have been called first.
        Returns {feature: score_drop} sorted descending.
        """
        self._require_fit()
        if self._train_result is None:
            raise RuntimeError("No trained model found. Call train() or train_auto() first.")
        if not self._target:
            raise ValueError("A target column is required.")

        from sklearn.model_selection import train_test_split
        from kaizenstat.utils.helpers import detect_task_type

        df = self._active_df.copy()
        df = df.loc[df[self._target].notna()]
        X  = df.drop(columns=[self._target])
        y  = df[self._target]

        task = detect_task_type(y)
        le   = self._train_result.label_encoder
        if le is not None:
            y = pd.Series(le.transform(y), index=y.index, name=self._target)

        try:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2,
                stratify=y if task == "classification" else None,
                random_state=42,
            )
        except ValueError:
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return self._debugger.feature_impact(
            self._train_result.pipeline, X_test, y_test, top_n=top_n
        )

    def recommend_actions(self) -> list:
        """
        Generate a prioritised what-to-do list based on the data profile and debug result.

        Requires fit(), train() (or train_auto()), and debug_model() to have been called.
        """
        self._require_fit()
        if self._debug_result is None:
            raise RuntimeError("No debug result. Call debug_model() first.")
        if not self._target:
            raise ValueError("A target column is required.")

        from kaizenstat.model.trainer import ModelTrainer

        df = self._active_df.copy()
        df = df.loc[df[self._target].notna()]
        X  = df.drop(columns=[self._target])
        y  = df[self._target]

        profile = ModelTrainer._analyze_data(X, y)
        return self._debugger.recommend_actions(profile, self._debug_result)

    # ------------------------------------------------------------------ #
    # Reliability / Trust                                                 #
    # ------------------------------------------------------------------ #

    def trust_score(self, test_size: float = 0.2) -> "TrustReport":
        """
        Compute a production-readiness Trust Report: confidence distribution,
        prediction uncertainty, robustness under perturbation, calibration gap,
        and failure-case slices. Works in both tabular and text mode.

        Reuses the exact held-out split from debug_model() when available.
        """
        self._require_fit()
        if self._train_result is None:
            console.print("[dim]Running train() first...[/dim]")
            self.train(test_size=test_size)

        if self._last_split is not None:
            _, X_test, _, y_test, task = self._last_split
        else:
            from sklearn.model_selection import train_test_split
            df = self._active_df.copy()
            df = df.loc[df[self._target].notna()].copy()
            y = df[self._target]
            task = detect_task_type(y)
            X = df[self._text_col].fillna("").astype(str) if self._mode == "text" \
                else df.drop(columns=[self._target])
            le = self._train_result.label_encoder
            if le is not None:
                y = pd.Series(le.transform(y), index=y.index, name=self._target)
            try:
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=test_size,
                    stratify=y if task == "classification" else None,
                    random_state=42,
                )
            except ValueError:
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

        return self._trust.analyze(self._train_result.pipeline, X_test, y_test, task=task)

    # ------------------------------------------------------------------ #

    def auto_improve_text(self, tune: bool = True) -> "ComparisonResult":
        """
        Text self-healing loop: baseline train → debug → apply NLP improvements
        (clean noise, prune rare tokens, char n-grams) → retrain → compare.

        Only valid in text mode. Returns a ComparisonResult with score_delta.
        """
        self._require_fit()
        if self._mode != "text":
            raise RuntimeError("auto_improve_text() requires a text dataset. "
                               "Use auto_improve() for tabular data.")

        console.print(Panel.fit("[bold cyan]Step 1/4[/bold cyan] — Baseline text model",
                                border_style="cyan"))
        baseline = self._text_trainer.train_best(
            self._active_df, self._target, text_col=self._text_col, tune=False,
        )

        console.print(Panel.fit("[bold cyan]Step 2/4[/bold cyan] — Diagnosing failure modes",
                                border_style="cyan"))
        self._train_result = baseline
        self.debug_model()

        console.print(Panel.fit("[bold cyan]Step 3/4[/bold cyan] — Cleaning text + pruning rare tokens",
                                border_style="cyan"))
        healed = self._heal_text(self._active_df)

        console.print(Panel.fit(
            "[bold cyan]Step 4/4[/bold cyan] — Retraining on healed text"
            + (" with tuning" if tune else ""),
            border_style="cyan"))
        improved = self._text_trainer.train_best(
            healed, self._target, text_col=self._text_col, tune=tune,
        )
        self._train_result = improved
        self._fixed_df = healed

        result = ComparisonResult(before=baseline, after=improved)
        result.display()
        return result

    def _heal_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply safe text cleaning to the dominant text column."""
        import re as _re
        out = df.copy()
        col = self._text_col
        url = _re.compile(r"https?://\S+|www\.\S+")
        html = _re.compile(r"<[^>]+>")
        multi_ws = _re.compile(r"\s+")

        def clean(t: str) -> str:
            t = str(t)
            t = url.sub(" ", t)
            t = html.sub(" ", t)
            t = multi_ws.sub(" ", t).strip()
            return t

        out[col] = out[col].fillna("").astype(str).apply(clean)
        # Drop empty/near-empty docs (keep target alignment)
        keep = out[col].str.split().str.len() >= 1
        removed = int((~keep).sum())
        if removed:
            console.print(f"[dim]Removed {removed} empty document(s) after cleaning[/dim]")
            out = out.loc[keep].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------ #

    def auto_improve(self, tune: bool = True) -> "ComparisonResult":
        """
        Detect issues, apply safe fixes, retrain, and compare before vs after.

        Returns a ComparisonResult with score_delta showing the improvement.
        """
        self._require_fit()

        console.print(Panel.fit(
            "[bold cyan]Step 1/3[/bold cyan] — Baseline training",
            border_style="cyan",
        ))
        if self._train_result is None:
            self.train(tune=False)
        baseline = self._train_result

        console.print(Panel.fit(
            "[bold cyan]Step 2/3[/bold cyan] — Applying safe data fixes",
            border_style="cyan",
        ))
        self.fix(safe=True)

        console.print(Panel.fit(
            f"[bold cyan]Step 3/3[/bold cyan] — Retraining"
            + (" with hyperparameter tuning" if tune else ""),
            border_style="cyan",
        ))
        self._train_result = None  # force retrain on fixed data
        self.train(tune=tune)
        improved = self._train_result

        result = ComparisonResult(before=baseline, after=improved)
        result.display()
        return result

    # ------------------------------------------------------------------ #
    # 10. pipeline_confidence                                              #
    # ------------------------------------------------------------------ #

    def pipeline_confidence(self) -> int:
        """
        Compute a 0–100 Pipeline Confidence Score based on health, validation,
        model stability, and test performance.
        """
        self._require_fit()
        score = 50

        if self._health_result is not None:
            score += int(self._health_result.score * 0.25)

        if self._validation_result is not None:
            n_issues = len(self._validation_result.issues)
            score -= min(20, n_issues * 5)

        if self._train_result is not None:
            score += int(self._train_result.test_score * 20)

        if self._debug_result is not None:
            gap = abs(self._debug_result.gap)
            score -= min(20, int(gap * 100))

        score = max(0, min(100, score))
        color = "green" if score >= 80 else ("yellow" if score >= 60 else "red")
        grade = "production-ready" if score >= 80 else ("needs work" if score >= 60 else "not ready")
        console.print(Panel.fit(
            f"[bold {color}]{score}% — {grade}[/bold {color}]",
            title="[bold]KaizenStat · Pipeline Confidence Score[/bold]",
            border_style=color,
        ))
        return score

    # ------------------------------------------------------------------ #
    # Plugin API                                                           #
    # ------------------------------------------------------------------ #

    def add_model(self, name: str, model: Any) -> "DataDoctor":
        """Register a custom model to include in the benchmark.

        Example::

            from sklearn.svm import SVC
            doctor.add_model("SVM", SVC(probability=True))
        """
        self._custom_models[name] = model
        console.print(f"[green]✓ Model '[bold]{name}[/bold]' registered — will compete in next benchmark.[/green]")
        return self

    def add_check(self, check_fn: Callable, name: str = "") -> "DataDoctor":
        """Register a custom validation check function.

        The function receives ``(df, target)`` and should return a list of
        issue strings (empty list = no issues).

        Example::

            def my_check(df, target):
                if df[target].nunique() < 2:
                    return ["Target has fewer than 2 classes"]
                return []

            doctor.add_check(my_check, name="target_classes")
        """
        label = name or getattr(check_fn, "__name__", "custom_check")
        self._custom_checks.append((label, check_fn))
        console.print(f"[green]✓ Check '[bold]{label}[/bold]' registered — will run in next validate().[/green]")
        return self

    # ------------------------------------------------------------------ #
    # Convenience pass-throughs                                            #
    # ------------------------------------------------------------------ #

    def export_model(self, path: str = "model.joblib") -> str:
        """Export the trained model pipeline to disk."""
        if self._train_result is None:
            raise RuntimeError("No trained model found. Call train() first.")
        return self._reporter.export_model(self._train_result.pipeline, path=path)

    def codegen(self, output_path: str = "pipeline.py") -> str:
        """Generate a standalone Python script reproducing the pipeline."""
        self._require_fit()
        task = ""
        if self._target and self._train_result:
            task = self._train_result.task
        return self._reporter.codegen(
            self._active_df, self._target, output_path=output_path, task=task
        )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    @property
    def _active_df(self) -> pd.DataFrame:
        return self._fixed_df if self._fixed_df is not None else self._df

    def _require_fit(self) -> None:
        if self._df is None:
            raise RuntimeError(
                "DataDoctor has not been fitted. Call doctor.fit(df, target='...') first."
            )

    def __repr__(self) -> str:
        fitted = self._df is not None
        target = f", target='{self._target}'" if self._target else ""
        shape = f", shape={self._df.shape}" if fitted else ""
        return f"DataDoctor(fitted={fitted}{shape}{target})"


# ---------------------------------------------------------------------------- #
# Module-level helpers                                                          #
# ---------------------------------------------------------------------------- #

def _detect_suspicious_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """
    Identify columns that will likely break or degrade the ML pipeline:
    - Near-unique ID columns (PassengerId, row numbers, UUIDs)
    - High-cardinality free-text with many words (Name, address, comments)

    These are safe to drop before training — they carry no generalizable signal.
    """
    import numpy as np

    suspicious: List[str] = []
    n = len(df)
    if n < 10:
        return suspicious

    # Common ID name patterns (case-insensitive)
    _ID_NAMES = {
        "passengerid", "id", "uuid", "guid", "rowid", "row_id",
        "index", "serial", "record_id", "customerid", "userid", "user_id",
        "order_id", "orderid", "ticket", "ticket_id",
    }
    # Common free-text name patterns
    _TEXT_NAMES = {"name", "fullname", "full_name", "firstname", "lastname",
                   "first_name", "last_name", "description", "notes", "comment",
                   "comments", "address", "street", "cabin"}

    for col in df.columns:
        if col in exclude:
            continue
        col_lower = col.lower().strip()
        s = df[col].dropna()
        if s.empty:
            continue

        # --- ID detection: near-unique integer or string column ---
        is_numeric_id = (
            np.issubdtype(df[col].dtype, np.integer)
            and s.nunique() > n * 0.9
        )
        is_string_id = (
            df[col].dtype == object
            and s.nunique() > n * 0.9
            and float(s.astype(str).str.split().str.len().mean()) <= 1.5
        )
        named_id = col_lower in _ID_NAMES or col_lower.endswith("_id") or col_lower.endswith("id")

        if is_numeric_id or is_string_id or (named_id and s.nunique() > n * 0.8):
            suspicious.append(col)
            continue

        # --- Free-text detection: object column with many words per row ---
        if df[col].dtype == object:
            avg_words = float(s.astype(str).str.split().str.len().mean())
            if col_lower in _TEXT_NAMES:
                suspicious.append(col)
                continue
            # High-cardinality + multi-word = free text (e.g. "Braund, Mr. Owen Harris")
            if avg_words >= 2.5 and s.nunique() > n * 0.7:
                suspicious.append(col)

    return suspicious
