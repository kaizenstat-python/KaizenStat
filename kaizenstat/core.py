# =========================================
# KAIZENSTAT 0.2.0 – NEURAL ENGINE (FINAL)
# =========================================

import os
import time
import warnings
import json
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

class SilentStr(str):
    """A string subclass that returns an empty string for __repr__ to avoid printing in REPLs/Jupyter."""
    def __repr__(self):
        return ""

class SilentDataFrame(pd.DataFrame):
    """A pandas DataFrame subclass that does not display itself in REPLs/Jupyter."""
    def _ipython_display_(self):
        pass
    def _repr_mimebundle_(self, include=None, exclude=None):
        return {}
    def __repr__(self):
        return ""
    def _repr_html_(self):
        return ""

def section(title):
    console.print(
        Panel.fit(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan"
        )
    )


import joblib

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# Optional engines
try:
    import polars as pl
    POLARS = True
except ImportError:
    POLARS = False

try:
    import xgboost as xgb
    XGB = True
except ImportError:
    XGB = False


# =========================================
# 🧠 HARDWARE DETECTION
# =========================================
def detect_device() -> str:
    """Detect the best available compute device (cuda, mps, or cpu)."""
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass
    return device


# =========================================
# ⚡ DATA ENGINE
# =========================================
class DataEngine:
    """Handles data loading with optional Polars acceleration."""

    @staticmethod
    def load(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from a CSV path or accept a DataFrame directly.

        Args:
            data: A file path (str) to a CSV or an existing pandas DataFrame.

        Returns:
            A pandas DataFrame copy of the input data.
        """
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"File not found: '{data}'")

            if POLARS:
                try:
                    df = pl.read_csv(data)
                    print("⚡ Loaded with Polars Engine")
                    return df.to_pandas()
                except Exception:
                    print("⚠️ Polars failed → fallback to Pandas")

            print("🐼 Loaded with Pandas Engine")
            return pd.read_csv(data)

        elif isinstance(data, pd.DataFrame):
            return data.copy()

        else:
            raise ValueError(f"Unsupported data input type: {type(data)}. Pass a CSV path or DataFrame.")


# =========================================
# 🔧 INTERNAL UTILITIES
# =========================================
def _detect_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that look like row IDs or unique identifiers.
    Checks for: column name patterns and extreme uniqueness ratios.
    """
    id_patterns = ["id", "uuid", "index", "key", "serial", "row_num"]
    id_cols = []
    for col in df.columns:
        col_lower = col.lower().strip()
        # Name-based detection
        if col_lower in id_patterns or col_lower.endswith("_id") or col_lower.startswith("id_"):
            id_cols.append(col)
            continue
        # Uniqueness-based detection: if a string/int column has >95% unique values
        if df[col].dtype == "object" or np.issubdtype(df[col].dtype, np.integer):
            if df[col].nunique() > len(df) * 0.95 and len(df) > 20:
                id_cols.append(col)
    return id_cols


def _detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are datetime or look like datetime strings."""
    dt_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
            continue
        if df[col].dtype == "object":
            sample = df[col].dropna().head(20)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    dt_cols.append(col)
                except (ValueError, TypeError):
                    pass
    return dt_cols


def _detect_imbalance(y: pd.Series) -> bool:
    """Returns True if the majority class makes up > 65% of the target."""
    counts = y.value_counts(normalize=True)
    return counts.iloc[0] > 0.65 if len(counts) > 1 else False


def _detect_task_type(y: pd.Series) -> bool:
    """
    Detect whether the target is classification or regression.
    Returns True for classification, False for regression.
    """
    # String or category → always classification
    if y.dtype == "object" or y.dtype.name == "category":
        return True
    # Float → always regression (even if few unique values)
    if np.issubdtype(y.dtype, np.floating):
        return False
    # Integer → classification only if few unique values
    if np.issubdtype(y.dtype, np.integer) and y.nunique() <= 50:
        return True
    return False


# =========================================
# 🚀 KAIZENSTAT CORE
# =========================================
class KaizenStat:
    """
    KAIZENSTAT v0.2.0: Zero-friction AutoML + Data Cleaning Toolkit.

    Methods:
        audit(df, target)        → Diagnostic sweep
        heal(df, target)         → Auto-clean dataset
        benchmark(df, target)    → Train & rank models
        auto(data, target)       → Full pipeline in one call
        explain(data, target)    → Plain-English summary
        codegen(data, target, output_path) → Generate standalone Python script
        report(data, target, output_path)  → Generate interactive HTML report
        save_model(pipeline, path) → Export trained model
        load_model(path)         → Load exported model
        analyze(df, target)      → Intelligent dataset analysis
        ask(query)               → Conversational AI support
        ask_followup(query)      → Conversational AI follow-up support
    """
    DEFAULT_API_KEY = None  # API key should be set via environment variable or passed at runtime
    _last_context = None
    _conversation_history = []

    # ==========================
    # 🧠 VALIDATION
    # ==========================
    @staticmethod
    def _validate_df(df: pd.DataFrame, target: Optional[str] = None) -> None:
        """Validate the DataFrame and target column for common issues."""
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Dataset is empty")

        if target:
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
            if df[target].dropna().empty:
                raise ValueError("Target column contains only NaN values")

        if len(df) < 2:
            raise ValueError("Dataset too small for modeling (need at least 2 rows)")

    # ==========================
    # 🔍 AUDIT
    # ==========================
    @staticmethod
    def audit(df: pd.DataFrame, target: Optional[str] = None) -> Dict:
        """
        Run a comprehensive diagnostic sweep over the dataset.

        Args:
            df: The input DataFrame.
            target: Optional name of the target column.

        Returns:
            A dictionary containing all audit findings.
        """
        KaizenStat._validate_df(df, target)

        section("KAIZENSTAT AUDIT")

        findings = {
            "shape": df.shape,
            "duplicates": int(df.duplicated().sum()),
            "missing_values": int(df.isna().sum().sum()),
        }

        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
        table.add_row("Duplicates", str(findings['duplicates']))
        table.add_row("Missing Values", str(findings['missing_values']))

        num_df = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.values).sum()) if not num_df.empty else 0
        findings["infinite_values"] = inf_count
        table.add_row("Infinite Values", str(inf_count))

        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        findings["constant_columns"] = constant_cols
        table.add_row("Constant Columns", str(constant_cols) if constant_cols else 'None')

        id_cols = _detect_id_columns(df)
        findings["id_columns"] = id_cols
        if id_cols:
            table.add_row("ID-like Columns", f"[yellow]⚠️ {id_cols}[/yellow]")

        dt_cols = _detect_datetime_columns(df)
        findings["datetime_columns"] = dt_cols
        if dt_cols:
            table.add_row("Datetime Columns", f"[yellow]⚠️ {dt_cols}[/yellow]")

        if target:
            target_missing = int(df[target].isna().sum())
            findings["target_missing"] = target_missing
            table.add_row("Target Missing", str(target_missing))

            task_type = "Classification" if _detect_task_type(df[target].dropna()) else "Regression"
            findings["task_type"] = task_type
            table.add_row("Detected Task", f"[bold green]{task_type}[/bold green]")

            if task_type == "Classification":
                imbalanced = _detect_imbalance(df[target].dropna())
                findings["imbalanced"] = imbalanced
                if imbalanced:
                    table.add_row("Class Imbalance", "[yellow]⚠️ Detected (majority > 65%)[/yellow]")

        console.print(table)
        KaizenStat._last_audit_findings = findings
        return findings

    # ==========================
    # 🩹 HEAL
    # ==========================
    @staticmethod
    def heal(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """
        Automatically clean and repair a dataset.

        Actions performed:
            - Drop rows with missing target values
            - Remove duplicate rows
            - Replace infinite values with NaN
            - Drop columns that are >90% missing
            - Drop constant (zero-variance) columns
            - Drop detected ID and datetime columns
            - Impute numeric NaNs with median
            - Impute categorical NaNs with mode

        Args:
            df: The input DataFrame.
            target: Optional name of the target column.

        Returns:
            A cleaned copy of the DataFrame.
        """
        KaizenStat._validate_df(df, target)

        df = df.copy()
        actions = []

        # Drop missing target rows
        if target and target in df.columns:
            before = len(df)
            df = df.dropna(subset=[target])
            dropped_target = before - len(df)
            if dropped_target > 0:
                actions.append(f"Dropped {dropped_target} rows with missing target '{target}'")

        # Deduplicate
        before = len(df)
        df = df.drop_duplicates()
        deduped = before - len(df)
        if deduped > 0:
            actions.append(f"Removed {deduped} duplicate rows")

        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop bad columns
        dropped_cols = []
        for col in list(df.columns):
            if col == target:
                continue
            if df[col].isna().mean() > 0.9:
                df.drop(columns=[col], inplace=True)
                dropped_cols.append((col, "90%+ missing"))
            elif df[col].nunique() <= 1:
                df.drop(columns=[col], inplace=True)
                dropped_cols.append((col, "constant"))

        # Drop ID columns
        id_cols = [c for c in _detect_id_columns(df) if c != target and c in df.columns]
        if id_cols:
            df.drop(columns=id_cols, inplace=True)
            for c in id_cols:
                dropped_cols.append((c, "ID-like column"))

        # Drop datetime columns
        dt_cols = [c for c in _detect_datetime_columns(df) if c != target and c in df.columns]
        if dt_cols:
            df.drop(columns=dt_cols, inplace=True)
            for c in dt_cols:
                dropped_cols.append((c, "datetime column"))

        # Impute numeric
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        if num_cols:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            actions.append(f"Filled {len(num_cols)} numeric columns with median")

        # Impute categorical
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)
        for col in cat_cols:
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
        if cat_cols:
            actions.append(f"Filled {len(cat_cols)} categorical columns with mode")

        # Print report
        section("🩹 KAIZENSTAT HEAL")
        table = Table(box=box.ROUNDED)
        table.add_column("Status", style="bold")
        table.add_column("Action", style="white")

        if dropped_cols:
            for col, reason in dropped_cols:
                table.add_row("[bold bright_red]✗ Dropped[/bold bright_red]", f"'{col}' ({reason})")
        for action in actions:
            table.add_row("[green]✓ Fixed[/green]", action)
        if not dropped_cols and not actions:
            table.add_row("[green]✓ Perfect[/green]", "Dataset was already clean")

        console.print(table)
        KaizenStat._last_dropped_cols = dropped_cols
        return df

    # ==========================
    # 🚀 BENCHMARK
    # ==========================
    @staticmethod
    def benchmark(df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Auto-detect task type, build preprocessing pipelines,
        and rank models using cross-validation.

        Args:
            df: A cleaned DataFrame.
            target: Name of the target column.

        Returns:
            A DataFrame leaderboard sorted by score (descending).
        """
        KaizenStat._validate_df(df, target)

        section("🚀 KAIZENSTAT BENCHMARK")
        device = detect_device()
        section(f"⚡ RUNNING ON: {device.upper()}")

        X = df.drop(columns=[target])
        y = df[target].copy()

        # Convert numeric-like strings
        X = X.apply(pd.to_numeric, errors='ignore')

        if X.shape[1] == 0:
            raise ValueError("No feature columns available after dropping target")

        # Detect task type
        is_classification = _detect_task_type(y)
        task_str = "CLASSIFICATION" if is_classification else "REGRESSION"
        console.print(f"  [bold]Task:[/] [green]{task_str}[/green]")

        # Encode string targets for classification
        label_encoder = None
        if is_classification and y.dtype == "object":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            mapping_str = ", ".join(f"'{cls}': {int(idx)}" for cls, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            console.print(f"  [green]✓ Encoded target labels:[/] [bold cyan]{{{mapping_str}}}[/bold cyan]")

        # Feature type detection
        num_features = X.select_dtypes(include=[np.number]).columns
        cat_features = X.select_dtypes(exclude=[np.number]).columns

        if len(num_features) == 0 and len(cat_features) == 0:
            raise ValueError("No usable features found")

        # Preprocessor
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False), cat_features)
        ])

        # Imbalance detection
        imbalanced = _detect_imbalance(y) if is_classification else False
        if imbalanced:
            console.print("  [yellow]⚠️ Imbalanced classes → applying class_weight='balanced'[/yellow]")

        # Smart model selection
        n_rows = len(df)
        cat_dominant = len(cat_features) > len(num_features)
        models = {}

        if is_classification:
            weight_param = {"class_weight": "balanced"} if imbalanced else {}
            models["Logistic"] = LogisticRegression(max_iter=1000, **weight_param)
            models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42, **weight_param)
            if n_rows <= 100000:
                models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            models["Ridge"] = Ridge()
            models["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=42)
            if n_rows <= 100000:
                models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # XGBoost GPU support
        if XGB:
            try:
                if is_classification:
                    models["XGBoost"] = xgb.XGBClassifier(
                        tree_method="hist",
                        device=device if device == "cuda" else "cpu",
                        random_state=42
                    )
                else:
                    models["XGBoost"] = xgb.XGBRegressor(
                        tree_method="hist",
                        device=device if device == "cuda" else "cpu",
                        random_state=42
                    )
            except Exception:
                pass

        if cat_dominant:
            console.print("  [cyan]ℹ️ Categorical-heavy dataset → tree models preferred[/cyan]")

        # Adaptive CV
        cv = 2 if n_rows > 100000 else min(5, max(2, n_rows))

        # Scoring metric
        scoring = "accuracy" if is_classification else "r2"

        results = []
        best_pipeline = None
        best_score = -np.inf

        for name, model in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])

            from rich.progress import Progress, SpinnerColumn, TextColumn
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]Training {name}...[/cyan]"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task("", total=None)
                try:
                    start = time.time()
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                    latency = time.time() - start
                    score = round(float(scores.mean()), 4)
                    std = round(float(scores.std()), 4)

                    results.append({
                        "Model": name,
                        "Score": score,
                        "Std": std,
                        "Time(s)": round(latency, 3)
                    })

                    if score > best_score:
                        best_score = score
                        # Fit on full data for potential export
                        pipe.fit(X, y)
                        best_pipeline = pipe

                except Exception as e:
                    results.append({
                        "Model": name,
                        "Score": 0.0,
                        "Std": 0.0,
                        "Time(s)": 0.0,
                        "Error": str(e)[:60]
                    })

        results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)

        # Print leaderboard
        metric_name = "Accuracy" if is_classification else "R² Score"
        console.print(f"\n🏆 [bold cyan]Model Leaderboard ranked by {metric_name}:[/bold cyan]")
        table = Table(box=box.ROUNDED)
        table.add_column("Rank", justify="center")
        table.add_column("Model", style="cyan")
        table.add_column("Score", style="bold green")
        table.add_column("Std Dev", style="dim")
        table.add_column("Time(s)", style="yellow")
        
        for idx, row in results_df.iterrows():
            medal = "🥇" if idx == 0 else ("🥈" if idx == 1 else "🥉" if idx == 2 else f"#{idx+1}")
            table.add_row(
                medal,
                row['Model'],
                f"{row['Score']:.4f}",
                f"± {row['Std']:.4f}",
                f"{row['Time(s)']:.3f}s"
            )
        console.print(table)

        # Store best pipeline as an attribute for export
        KaizenStat._last_pipeline = best_pipeline
        KaizenStat._last_label_encoder = label_encoder
        KaizenStat._last_task_type = "classification" if is_classification else "regression"
        KaizenStat._last_target = target
        KaizenStat._last_results_df = results_df

        return SilentDataFrame(results_df)

    # ==========================
    # 🧠 AUTO MODE
    # ==========================
    @staticmethod
    def auto(data: Union[str, pd.DataFrame], target: str) -> pd.DataFrame:
        """
        Full pipeline: load → audit → heal → benchmark in a single call.

        Args:
            data: CSV path or DataFrame.
            target: Name of the target column.

        Returns:
            A DataFrame leaderboard of model results.
        """
        print("\n🚀 KAIZENSTAT 0.2.0 – NEURAL ENGINE\n")

        df = DataEngine.load(data)
        KaizenStat.audit(df, target)
        df = KaizenStat.heal(df, target)
        KaizenStat._validate_df(df, target)
        results = KaizenStat.benchmark(df, target)

        print(f"\n🏆 BEST MODEL: {results.iloc[0]['Model']} (Score: {results.iloc[0]['Score']:.4f})")
        
        # Build and store context for conversational AI
        KaizenStat._last_context = KaizenStat._build_context(df, target)

        return SilentDataFrame(results)

    # ==========================
    # 💬 EXPLAIN
    # ==========================
    @staticmethod
    def explain(data: Union[str, pd.DataFrame], target: str) -> str:
        """
        Run the full pipeline and generate a plain-English executive summary.

        Args:
            data: CSV path or DataFrame.
            target: Name of the target column.

        Returns:
            A formatted string containing the explanation report.
        """
        df = DataEngine.load(data)

        # Audit phase
        findings = KaizenStat.audit(df, target)

        # Heal phase
        df_clean = KaizenStat.heal(df, target)

        # Benchmark phase
        results = KaizenStat.benchmark(df_clean, target)

        # Dataset details
        rows, cols = findings["shape"]
        task = findings.get("task_type", "Unknown")

        # Build visual presentation using rich
        from rich.console import Console
        from rich.panel import Panel
        
        summary_lines = [
            f"📊 [bold cyan]DATASET SUMMARY[/bold cyan]",
            f"  • Dataset contains [bold green]{rows:,}[/bold green] rows and [bold green]{cols}[/bold green] columns.",
            f"  • Target column is '[bold green]{target}[/bold green]' (detected task: [bold green]{task}[/bold green])."
        ]

        issues_lines = [
            f"\n🔎 [bold cyan]ISSUES SWEEP & AUTO-HEALING[/bold cyan]"
        ]
        issues_detected = []
        if findings["missing_values"] > 0:
            issues_detected.append(f"  • [bold bright_yellow]Missing Values:[/] {findings['missing_values']:,} missing cells were detected and filled using median/mode.")
        if findings["duplicates"] > 0:
            issues_detected.append(f"  • [bold bright_yellow]Duplicate Rows:[/] {findings['duplicates']:,} duplicate rows were removed.")
        if findings["infinite_values"] > 0:
            issues_detected.append(f"  • [bold bright_yellow]Infinite values:[/] {findings['infinite_values']:,} infinite values were replaced.")
        if findings.get("id_columns"):
            issues_detected.append(f"  • [bold bright_red]ID Columns:[/] String identifiers {findings['id_columns']} were dropped to prevent leakage.")
        if findings.get("datetime_columns"):
            issues_detected.append(f"  • [bold bright_red]Datetime Columns:[/] Datetime values {findings['datetime_columns']} were removed.")
        if findings.get("constant_columns"):
            issues_detected.append(f"  • [bold bright_red]Constant Columns:[/] Uninformative columns {findings['constant_columns']} were dropped.")
        if findings.get("imbalanced"):
            issues_detected.append("  • [bold bright_yellow]Class Imbalance:[/] Skewed distribution detected. Adaptive class weighting has been applied.")
        if not issues_detected:
            issues_detected.append("  • [bold green]✓ Clean Slate:[/] No critical quality issues were found in the dataset.")
        issues_lines.extend(issues_detected)

        best = results.iloc[0]
        results_lines = [
            f"\n🤖 [bold cyan]MODEL RESULTS & LEADERBOARD[/bold cyan]",
            f"  🥇 [bold green]Winner:[/] [bold cyan]{best['Model']}[/bold cyan] with a cross-validated score of [bold green]{best['Score']:.4f}[/bold green]"
        ]

        if len(results) > 1:
            runner = results.iloc[1]
            results_lines.append(f"  🥈 [bold]Runner-up:[/] [bold cyan]{runner['Model']}[/bold cyan] with a score of [bold green]{runner['Score']:.4f}[/bold green]")

            if runner["Time(s)"] > 0 and best["Time(s)"] > 0:
                speed_ratio = best["Time(s)"] / runner["Time(s)"]
                if speed_ratio > 2:
                    results_lines.append(f"\n  💡 [bold bright_yellow]Speed Advantage:[/] {runner['Model']} is {speed_ratio:.1f}x faster to train than {best['Model']}.")
                    results_lines.append(f"     If latency or compute budget is constrained, consider using {runner['Model']}.")

        results_lines.append(f"\n📌 [bold cyan]RECOMMENDATION[/bold cyan]\n  Deploy model [bold green]'{best['Model']}'[/bold green] for the best generalization accuracy.")

        content = "\n".join(summary_lines + issues_lines + results_lines)
        console.print()
        console.print(Panel(content, title="[bold green]💡 KAIZENSTAT EXPLAINER REPORT[/bold green]", border_style="cyan"))

        plain_summary = f"KaizenStat Explainer Report: Dataset has {rows} rows, {cols} cols. Winner: {best['Model']}"
        return SilentStr(plain_summary)

    # ==========================
    # 📝 CODEGEN
    # ==========================
    @staticmethod
    def codegen(data: Union[str, pd.DataFrame], target: str, output_path: str = "pipeline.py") -> str:
        """
        Generate a standalone, dependency-free Python script that reproduces
        the exact data cleaning and model training pipeline.

        Args:
            data: CSV path or DataFrame.
            target: Name of the target column.
            output_path: Path to write the generated .py file.

        Returns:
            The path to the generated script.
        """
        df = DataEngine.load(data)
        data_path = data if isinstance(data, str) else "data.csv"
        if isinstance(data, pd.DataFrame):
            try:
                data.to_csv("data.csv", index=False)
            except Exception:
                pass

        # Run pipeline to determine best model
        df_clean = KaizenStat.heal(df, target)
        results = KaizenStat.benchmark(df_clean, target)
        best_model_name = results.iloc[0]["Model"]

        # Detect features
        is_classification = _detect_task_type(df_clean[target].dropna())
        num_features = list(df_clean.drop(columns=[target]).select_dtypes(include=[np.number]).columns)
        cat_features = list(df_clean.drop(columns=[target]).select_dtypes(exclude=[np.number]).columns)

        # Map model names to import strings and constructor strings
        model_map = {
            "Logistic": ("from sklearn.linear_model import LogisticRegression", "LogisticRegression(max_iter=1000)"),
            "Ridge": ("from sklearn.linear_model import Ridge", "Ridge()"),
            "RandomForest": (
                f"from sklearn.ensemble import {'RandomForestClassifier' if is_classification else 'RandomForestRegressor'}",
                f"{'RandomForestClassifier' if is_classification else 'RandomForestRegressor'}(n_estimators=100, random_state=42)"
            ),
            "GradientBoosting": (
                f"from sklearn.ensemble import {'GradientBoostingClassifier' if is_classification else 'GradientBoostingRegressor'}",
                f"{'GradientBoostingClassifier' if is_classification else 'GradientBoostingRegressor'}(n_estimators=100, random_state=42)"
            ),
            "XGBoost": ("import xgboost as xgb", f"xgb.{'XGBClassifier' if is_classification else 'XGBRegressor'}(random_state=42)"),
        }

        model_import, model_constructor = model_map.get(best_model_name, model_map["RandomForest"])

        needs_label_encoder = is_classification and df_clean[target].dtype == "object"

        code = f'''# ==========================================
# AUTO-GENERATED PIPELINE by KaizenStat
# Best Model: {best_model_name}
# Task: {"Classification" if is_classification else "Regression"}
# ==========================================

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
{"from sklearn.preprocessing import LabelEncoder" if needs_label_encoder else ""}
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
{model_import}
import joblib

# 1. Load Data
df = pd.read_csv("{data_path}")

# 2. Data Cleaning
df = df.dropna(subset=["{target}"])
df = df.drop_duplicates()
df = df.replace([np.inf, -np.inf], np.nan)

# Drop constant columns
for col in list(df.columns):
    if df[col].nunique() <= 1 and col != "{target}":
        df = df.drop(columns=[col])

# 3. Define Features and Target
num_features = {num_features}
cat_features = {cat_features}

X = df[num_features + cat_features].copy()
y = df["{target}"]
{"" if not needs_label_encoder else """
# Encode string labels
le = LabelEncoder()
y = le.fit_transform(y)
"""}
# Fill missing values
if num_features:
    X.loc[:, num_features] = X[num_features].fillna(X[num_features].median())
for col in cat_features:
    X.loc[:, col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown")

# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
])

# 5. Model Pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", {model_constructor})
])

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Model
pipeline.fit(X_train, y_train)

# 8. Evaluate
score = pipeline.score(X_test, y_test)
print(f"Model Score: {{score:.4f}}")

# 9. Save Model
joblib.dump(pipeline, "trained_pipeline.joblib")
print("Model saved to trained_pipeline.joblib")
'''

        with open(output_path, "w") as f:
            f.write(code)

        print(f"\n📝 Generated standalone pipeline script → {output_path}")
        return output_path

    # ==========================
    # 💾 MODEL EXPORT & LOAD
    # ==========================
    @staticmethod
    def save_model(pipeline=None, path: str = "model.joblib") -> str:
        """
        Save a trained pipeline to disk using joblib.

        Args:
            pipeline: A fitted sklearn Pipeline. If None, uses the last benchmarked pipeline.
            path: Output file path.

        Returns:
            The path where the model was saved.
        """
        if pipeline is None:
            pipeline = getattr(KaizenStat, "_last_pipeline", None)
        if pipeline is None:
            raise ValueError("No trained pipeline found. Run benchmark() or auto() first.")

        joblib.dump(pipeline, path)
        print(f"💾 Model saved → {path}")

        # Also save label encoder if one was used
        le = getattr(KaizenStat, "_last_label_encoder", None)
        if le is not None:
            le_path = path.replace(".joblib", "_label_encoder.joblib")
            joblib.dump(le, le_path)
            print(f"💾 Label encoder saved → {le_path}")

        return path

    @staticmethod
    def load_model(path: str = "model.joblib"):
        """
        Load a saved pipeline from disk.

        Args:
            path: Path to the .joblib file.

        Returns:
            The loaded sklearn Pipeline.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: '{path}'")

        pipeline = joblib.load(path)
        print(f"✓ Model loaded from {path}")
        return pipeline

    # ==========================
    # 📊 HTML REPORT
    # ==========================
    @staticmethod
    def report(data: Union[str, pd.DataFrame], target: str, output_path: str = "report.html", open_browser: bool = True) -> str:
        """
        Generate a rich, interactive HTML profiling report with Chart.js graphs.

        Args:
            data: CSV path or DataFrame.
            target: Name of the target column.
            output_path: Path to write the HTML report.
            open_browser: Whether to automatically open the report in the default web browser.

        Returns:
            The path to the generated HTML report.
        """
        df = DataEngine.load(data)

        # Audit
        findings = KaizenStat.audit(df, target)

        # Heal
        df_clean = KaizenStat.heal(df, target)

        # Benchmark
        results = KaizenStat.benchmark(df_clean, target)

        # Prepare chart data
        is_classification = _detect_task_type(df_clean[target].dropna())

        # Missing values per column
        missing_data = df.isna().sum()
        missing_labels = missing_data[missing_data > 0].index.tolist()
        missing_values = missing_data[missing_data > 0].values.tolist()

        # Target distribution
        if is_classification:
            target_counts = df_clean[target].value_counts()
            target_labels = [str(x) for x in target_counts.index.tolist()]
            target_values = target_counts.values.tolist()
        else:
            # Histogram bins for regression
            counts, bin_edges = np.histogram(df_clean[target].dropna(), bins=20)
            target_labels = [f"{bin_edges[i]:.1f}" for i in range(len(counts))]
            target_values = counts.tolist()

        # Correlation matrix (numeric only)
        num_df = df_clean.select_dtypes(include=[np.number])
        if len(num_df.columns) > 1:
            corr = num_df.corr()
            corr_labels = list(corr.columns)
            corr_data = corr.values.tolist()
        else:
            corr_labels = []
            corr_data = []

        # Model leaderboard
        model_names = results["Model"].tolist()
        model_scores = results["Score"].tolist()
        model_times = results["Time(s)"].tolist()

        # Column type breakdown
        n_numeric = len(df_clean.select_dtypes(include=[np.number]).columns)
        n_categorical = len(df_clean.select_dtypes(exclude=[np.number]).columns)

        rows, cols = findings["shape"]
        task_type = "Classification" if is_classification else "Regression"

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KaizenStat Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 40px;
            text-align: center;
            border-bottom: 3px solid #e94560;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(to right, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header p {{ color: #888; font-size: 1.1em; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #2a2a4a;
            text-align: center;
        }}
        .stat-card h3 {{ color: #888; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-card .value {{ font-size: 2em; font-weight: 700; color: #e94560; margin-top: 8px; }}
        .section {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 25px;
            border: 1px solid #2a2a4a;
        }}
        .section h2 {{
            color: #e94560;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 1px solid #2a2a4a;
            padding-bottom: 10px;
        }}
        .chart-container {{ position: relative; height: 350px; margin: 15px 0; }}
        .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }}
        @media (max-width: 768px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
        th {{ color: #e94560; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }}
        td {{ color: #ccc; }}
        tr:hover {{ background: #16213e; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .badge-success {{ background: #1b4332; color: #52b788; }}
        .badge-warn {{ background: #5c4813; color: #f0ad4e; }}
        .corr-table {{ font-size: 0.85em; overflow-x: auto; }}
        .corr-table td {{ text-align: center; padding: 8px; }}
        .footer {{ text-align: center; padding: 30px; color: #555; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 KaizenStat Report</h1>
        <p>Auto-Generated Data Profiling & Model Benchmarking Report</p>
    </div>

    <div class="container">
        <!-- Overview Cards -->
        <div class="grid">
            <div class="stat-card">
                <h3>Rows</h3>
                <div class="value">{rows:,}</div>
            </div>
            <div class="stat-card">
                <h3>Columns</h3>
                <div class="value">{cols}</div>
            </div>
            <div class="stat-card">
                <h3>Task Type</h3>
                <div class="value" style="font-size:1.4em">{task_type}</div>
            </div>
            <div class="stat-card">
                <h3>Best Score</h3>
                <div class="value">{model_scores[0]:.4f}</div>
            </div>
        </div>

        <!-- Data Quality -->
        <div class="section">
            <h2>📋 Data Quality Summary</h2>
            <div class="grid">
                <div class="stat-card">
                    <h3>Missing Values</h3>
                    <div class="value">{findings["missing_values"]:,}</div>
                </div>
                <div class="stat-card">
                    <h3>Duplicates</h3>
                    <div class="value">{findings["duplicates"]:,}</div>
                </div>
                <div class="stat-card">
                    <h3>Numeric Cols</h3>
                    <div class="value">{n_numeric}</div>
                </div>
                <div class="stat-card">
                    <h3>Categorical Cols</h3>
                    <div class="value">{n_categorical}</div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="chart-row">
            <div class="section">
                <h2>📊 Missing Values by Column</h2>
                <div class="chart-container">
                    <canvas id="missingChart"></canvas>
                </div>
            </div>
            <div class="section">
                <h2>🎯 Target Distribution</h2>
                <div class="chart-container">
                    <canvas id="targetChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Model Leaderboard -->
        <div class="section">
            <h2>🏆 Model Leaderboard</h2>
            <div class="chart-container">
                <canvas id="modelChart"></canvas>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'<tr><td>{"🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"}</td><td>{n}</td><td><span class="badge badge-success">{s:.4f}</span></td><td>{t:.3f}s</td></tr>' for i, (n, s, t) in enumerate(zip(model_names, model_scores, model_times)))}
                </tbody>
            </table>
        </div>

        <!-- Correlation Matrix -->
        {"" if not corr_labels else f"""
        <div class="section">
            <h2>🔗 Feature Correlation Matrix</h2>
            <div class="corr-table" style="overflow-x:auto;">
                <table>
                    <thead><tr><th></th>{"".join(f"<th>{c[:12]}</th>" for c in corr_labels)}</tr></thead>
                    <tbody>
                    {"".join(
                        "<tr><td><strong>" + corr_labels[i][:12] + "</strong></td>" +
                        "".join(
                            f'<td style="background:rgba(233,69,96,{abs(corr_data[i][j])*0.6:.2f});color:#fff">{corr_data[i][j]:.2f}</td>'
                            for j in range(len(corr_labels))
                        ) + "</tr>"
                        for i in range(len(corr_labels))
                    )}
                    </tbody>
                </table>
            </div>
        </div>
        """}
    </div>

    <div class="footer">
        Generated by <a href="https://www.kaizenstat.com/" target="_blank" style="color: #48dbfb; text-decoration: none;">KaizenStat v0.2.13</a> | {time.strftime("%Y-%m-%d %H:%M:%S")}
    </div>

    <script>
        const chartColors = ['#e94560', '#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff', '#5f27cd', '#01a3a4'];

        // Missing Values Chart
        new Chart(document.getElementById('missingChart'), {{
            type: 'bar',
            data: {{
                labels: {missing_labels},
                datasets: [{{
                    label: 'Missing Count',
                    data: {missing_values},
                    backgroundColor: '#e94560',
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: '#2a2a4a' }}, ticks: {{ color: '#888' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#888', maxRotation: 45 }} }}
                }}
            }}
        }});

        // Target Distribution Chart
        new Chart(document.getElementById('targetChart'), {{
            type: '{"doughnut" if is_classification else "bar"}',
            data: {{
                labels: {target_labels},
                datasets: [{{
                    label: 'Count',
                    data: {target_values},
                    backgroundColor: chartColors.slice(0, {len(target_values)}),
                    {"borderRadius: 6," if not is_classification else ""}
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: '{"right" if is_classification else "top"}', labels: {{ color: '#ccc' }} }} }}
            }}
        }});

        // Model Scores Chart
        new Chart(document.getElementById('modelChart'), {{
            type: 'bar',
            data: {{
                labels: {model_names},
                datasets: [{{
                    label: 'Score',
                    data: {model_scores},
                    backgroundColor: chartColors.slice(0, {len(model_names)}),
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#2a2a4a' }}, ticks: {{ color: '#888' }}, min: 0, max: 1 }},
                    y: {{ grid: {{ display: false }}, ticks: {{ color: '#ccc' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

        with open(output_path, "w") as f:
            f.write(html)

        print(f"\n📊 HTML Report generated → {output_path}")

        if open_browser:
            # Try to display inline in Jupyter/Colab
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    from IPython.display import display, HTML
                    import base64
                    b64_html = base64.b64encode(html.encode('utf-8')).decode('utf-8')
                    iframe = f'<iframe src="data:text/html;base64,{b64_html}" width="100%" height="800px" style="border:1px solid #2a2a4a; border-radius:12px; margin-top: 15px;"></iframe>'
                    display(HTML(iframe))
                    return output_path
            except Exception:
                pass

            import webbrowser
            try:
                webbrowser.open("file://" + os.path.abspath(output_path))
            except Exception:
                pass

        return output_path

    @staticmethod
    def serve_report(output_path: str, port: Optional[int] = None) -> None:
        """
        Serve the generated HTML report locally on a temporary port.
        """
        import http.server
        import socketserver
        import socket
        import webbrowser

        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]

        directory = os.path.dirname(os.path.abspath(output_path))
        filename = os.path.basename(output_path)

        original_dir = os.getcwd()
        os.chdir(directory)

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                # Suppress log output to keep terminal clean
                pass

        try:
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("127.0.0.1", port), QuietHandler) as httpd:
                url = f"http://127.0.0.1:{port}/{filename}"
                console.print(f"\n🌐 Serving report at: [bold cyan]{url}[/bold cyan]")
                
                # Expose public link in Google Colab
                try:
                    import google.colab
                    colab_url = google.colab.kernel.proxy_port(port)
                    console.print(f"🔗 [bold green]Colab Public Link:[/] {colab_url}{filename}")
                except Exception:
                    pass

                console.print("[yellow]⚡ Press Ctrl+C to stop the server.[/yellow]")

                try:
                    webbrowser.open(url)
                except Exception:
                    pass

                httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Server stopped.[/yellow]")
        except Exception as e:
            console.print(f"\n[bold bright_red]✖ Error serving report:[/] {e}")
        finally:
            os.chdir(original_dir)

    # ==========================
    # 🌐 SERVE (Streamlit UI)
    # ==========================
    @staticmethod
    def serve(data: Union[str, pd.DataFrame], target: str, port: int = 8501) -> None:
        """
        Launch an interactive Streamlit web dashboard for the dataset.

        Args:
            data: CSV path or DataFrame.
            target: Name of the target column.
            port: Port to run the Streamlit server on.
        """
        data_path = data if isinstance(data, str) else None

        if data_path is None:
            # Save DataFrame to a temp file for Streamlit to read
            data_path = "_kaizenstat_temp_data.csv"
            data.to_csv(data_path, index=False)

        app_code = f'''
import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")

st.set_page_config(page_title="KaizenStat Dashboard", page_icon="🚀", layout="wide")

st.markdown("# 🚀 KaizenStat Interactive Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("{data_path}")

df = load_data()
target = "{target}"

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Audit", "🚀 Benchmark", "📁 Predict"])

with tab1:
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{{len(df):,}}")
    col2.metric("Columns", f"{{len(df.columns)}}")
    col3.metric("Missing", f"{{df.isna().sum().sum():,}}")
    col4.metric("Duplicates", f"{{df.duplicated().sum():,}}")

    st.markdown("### Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### Column Types")
    col_types = df.dtypes.value_counts().reset_index()
    col_types.columns = ["Type", "Count"]
    st.bar_chart(col_types.set_index("Type"))

    st.markdown("### Missing Values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.bar_chart(missing)
    else:
        st.success("No missing values!")

    st.markdown("### Target Distribution")
    if target in df.columns:
        st.bar_chart(df[target].value_counts().head(20))

with tab2:
    st.markdown("### 🔍 Audit Report")
    from kaizenstat import KaizenStat
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        findings = KaizenStat.audit(df, target)
    st.code(buf.getvalue())

    st.markdown("### Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        corr = num_df.corr()
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

with tab3:
    st.markdown("### 🚀 Model Benchmark")
    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Training models..."):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df_clean = KaizenStat.heal(df, target)
                results = KaizenStat.benchmark(df_clean, target)
            st.code(buf.getvalue())
            st.dataframe(results, use_container_width=True)
            st.bar_chart(results.set_index("Model")["Score"])

with tab4:
    st.markdown("### 📁 Upload New Data for Predictions")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.dataframe(new_df.head(), use_container_width=True)
        st.info("To predict, first run Benchmark to train a model, then use KaizenStat.load_model().")
'''

        app_file = "_kaizenstat_app.py"
        with open(app_file, "w") as f:
            f.write(app_code)

        print(f"\n🌐 Launching KaizenStat Dashboard on port {port}...")
        print(f"   Open: http://localhost:{port}")
        print(f"   Press Ctrl+C to stop\n")

        os.system(f"streamlit run {app_file} --server.port {port} --server.headless true")

    # ==========================
    # 🧠 AI CHAT & ANALYZE
    # ==========================
    @staticmethod
    def _build_context(df: pd.DataFrame, target: str) -> dict:
        # Check if we have pre-computed audit/heal info
        audit_findings = getattr(KaizenStat, "_last_audit_findings", {})
        if not audit_findings:
            # If not computed, run audit silently
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    audit_findings = KaizenStat.audit(df, target)
                except Exception:
                    audit_findings = {}

        # Calculate high cardinality columns
        high_card_cols = []
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if col != target:
                if df[col].nunique() > 20:
                    high_card_cols.append(col)

        # Get dropped columns
        dropped_cols = getattr(KaizenStat, "_last_dropped_cols", [])
        dropped_cols_list = []
        if isinstance(dropped_cols, list):
            for item in dropped_cols:
                if isinstance(item, tuple) and len(item) > 0:
                    dropped_cols_list.append(str(item[0]))
                else:
                    dropped_cols_list.append(str(item))

        # Get best model info
        best_model = "None"
        best_score = 0.0
        results_df = getattr(KaizenStat, "_last_results_df", None)
        if results_df is not None and not results_df.empty:
            best_model = results_df.iloc[0]["Model"]
            best_score = float(results_df.iloc[0]["Score"])

        # Class imbalance
        imbalance_detected = audit_findings.get("imbalanced", False)

        # Build missing columns detailed breakdown
        missing_counts = df.isna().sum()
        missing_dict = missing_counts[missing_counts > 0].to_dict()

        # Build issues list
        issues = []
        if imbalance_detected:
            issues.append("imbalance")
        if missing_dict:
            issues.append("missing_values")
        if high_card_cols:
            issues.append("high_cardinality")

        # Ensure all types in context are standard Python primitives for JSON serialization
        context = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "missing": {str(k): int(v) for k, v in missing_dict.items()},
            "dropped_cols": [str(c) for c in dropped_cols_list],
            "issues": issues,
            "model": str(best_model),
            "score": float(best_score)
        }

        # Save context to local file for CLI persistence
        try:
            with open(".kaizenstat_context.json", "w") as f:
                json.dump(context, f, indent=2)
        except Exception:
            pass

        return context

    @staticmethod
    def _get_system_prompt(context: dict) -> str:
        prompt_template = """You are KaizenStat AI — a senior data scientist embedded inside a data intelligence system.

You are NOT a generic chatbot.
You MUST ONLY use the structured dataset and pipeline context provided below.

========================================
📊 SYSTEM CONTEXT (SOURCE OF TRUTH)
========================================
{context}

This includes:
- dataset structure
- missing values
- dropped columns (and reasons)
- feature types
- detected issues (imbalance, leakage, etc.)
- model used
- model performance (score)
- preprocessing steps

========================================
🎯 YOUR OBJECTIVE
========================================

Your goal is to provide:
1. Deep, context-aware analysis
2. Practical, prioritized recommendations
3. Clear reasoning tied to THIS dataset only

DO NOT give generic advice.

========================================
🧠 THINK LIKE A SENIOR DATA SCIENTIST
========================================

When analyzing:

- Always explain WHY something is a problem
- Always connect issues to model performance
- Always prioritize the most impactful fix first
- Avoid listing everything — focus on what matters most

========================================
⚠️ IMPORTANT RULES
========================================

- DO NOT hallucinate missing information
- DO NOT assume columns that are not in context
- DO NOT give generic textbook advice
- DO NOT repeat the same point multiple times
- DO NOT say "it depends" without giving direction

========================================
📈 OUTPUT STYLE
========================================

If NO user question:

Return structured output:

🔍 Key Issues (max 3–5)
⚠️ Risks (why they matter)
🚀 Action Plan (ordered, most important first)

----------------------------------------

If user question exists:

💬 Direct Answer
🧠 Reasoning (based on context)
🚀 What to do next (clear steps)

----------------------------------------

========================================
🔥 INTELLIGENCE BOOST (VERY IMPORTANT)
========================================

- If model score is low → explain EXACT bottleneck
- If imbalance exists → prioritize fixing it FIRST
- If useful column was dropped → question it
- If feature quality is weak → suggest feature engineering
- If model is already good → suggest optimization, not random changes"""
        return prompt_template.replace("{context}", json.dumps(context, indent=2))

    @staticmethod
    def _build_ai_prompt(context: dict, user_query: Optional[str] = None) -> str:
        system_prompt = KaizenStat._get_system_prompt(context)
        if user_query:
            return f"{system_prompt}\n\n========================================\n👤 USER QUESTION\n========================================\n{user_query}"
        return system_prompt

    @staticmethod
    def _call_openrouter_api_messages(messages: list, api_key: Optional[str] = None) -> str:
        key = api_key or getattr(KaizenStat, "DEFAULT_API_KEY", "")
        if not key:
            raise ValueError("No OpenRouter API key found. Please provide one.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/masuddarrahaman/KaizenStat-Library",
            "X-Title": "KaizenStat Intelligence"
        }

        import ssl
        ssl_context = ssl._create_unverified_context()

        # Models list with fallback mechanisms
        models = [
            "google/gemini-2.5-flash",
            "meta-llama/llama-3-8b-instruct:free",
            "google/gemma-2-9b-it:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "google/gemini-2.5-pro"
        ]

        last_error = None
        for model in models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1500
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            try:
                # 15 seconds timeout
                with urllib.request.urlopen(req, context=ssl_context, timeout=15) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    if "choices" in res and len(res["choices"]) > 0:
                        return res["choices"][0]["message"]["content"]
            except HTTPError as e:
                err_body = e.read().decode("utf-8")
                try:
                    err_json = json.loads(err_body)
                    error_msg = err_json.get("error", {}).get("message", "")
                except Exception:
                    error_msg = err_body
                last_error = f"HTTP Error {e.code}: {error_msg}"
                print(f"⚠️ Model {model} failed or server busy: {last_error}. Trying fallback model...")
            except URLError as e:
                last_error = f"Network Error: {e.reason}"
                print(f"⚠️ Model {model} network error: {last_error}. Trying fallback model...")
            except Exception as e:
                last_error = f"Unexpected Error: {e}"
                print(f"⚠️ Model {model} failed: {last_error}. Trying fallback model...")

        raise RuntimeError(
            f"Failed to query OpenRouter. Last error: {last_error}\n"
            "Server might be busy or API token has expired. "
            "Please check your internet connection or try again. "
            "Alternatively, provide your own OpenRouter / Gemini API key via the `api_key` parameter."
        )

    @staticmethod
    def analyze(data: Union[str, pd.DataFrame], target: str, api_key: Optional[str] = None) -> str:
        """
        Perform auto-intelligence analysis on the dataset.

        Args:
            data: CSV path or DataFrame.
            target: Name of the target column.
            api_key: Optional custom OpenRouter API key.

        Returns:
            The plain-English structured analysis.
        """
        df = DataEngine.load(data)
        # Run auto pipeline to populate metrics
        KaizenStat.auto(df, target)

        context = KaizenStat._last_context
        prompt = KaizenStat._build_ai_prompt(context, user_query=None)

        print("\n🧠 Querying KaizenStat Intelligence Engine...")
        response = KaizenStat._call_openrouter_api_messages(
            [{"role": "user", "content": prompt}],
            api_key=api_key
        )

        # Initialize conversation history
        KaizenStat._conversation_history = [
            {"role": "user", "content": "Analyze this dataset."},
            {"role": "assistant", "content": response}
        ]

        # Persist conversation history to disk
        try:
            with open(".kaizenstat_history.json", "w") as f:
                json.dump(KaizenStat._conversation_history, f, indent=2)
        except Exception:
            pass

        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()
        console.print()
        console.print(Panel(Markdown(response), title="[bold green]🧠 KAIZENSTAT AUTOMATIC ANALYSIS[/]", border_style="cyan"))
        return SilentStr(response)

    @staticmethod
    def ask(user_query: str, api_key: Optional[str] = None) -> str:
        """
        Ask a conversational question about the last analyzed dataset context.

        Args:
            user_query: The question for the AI engine.
            api_key: Optional custom OpenRouter API key.

        Returns:
            The AI response.
        """
        # Load context from file if not present in memory (critical for CLI flow)
        if KaizenStat._last_context is None:
            try:
                with open(".kaizenstat_context.json", "r") as f:
                    KaizenStat._last_context = json.load(f)
            except Exception:
                pass

        context = KaizenStat._last_context
        if context is None:
            raise ValueError(
                "No dataset context found. Please run KaizenStat.analyze(df, target) "
                "or KaizenStat.auto(df, target) first."
            )

        prompt = KaizenStat._build_ai_prompt(context, user_query=user_query)

        print(f"\n🧠 Querying KaizenStat Intelligence for: '{user_query}'...")
        response = KaizenStat._call_openrouter_api_messages(
            [{"role": "user", "content": prompt}],
            api_key=api_key
        )

        # Reset history thread for this question
        KaizenStat._conversation_history = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response}
        ]

        # Persist conversation history to disk
        try:
            with open(".kaizenstat_history.json", "w") as f:
                json.dump(KaizenStat._conversation_history, f, indent=2)
        except Exception:
            pass

        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()
        console.print()
        console.print(Panel(Markdown(response), title="[bold green]💬 KAIZENSTAT RESPONSE[/]", border_style="cyan"))
        return SilentStr(response)

    @staticmethod
    def ask_followup(user_query: str, api_key: Optional[str] = None) -> str:
        """
        Ask a follow-up question keeping conversation history memory.

        Args:
            user_query: The follow-up question.
            api_key: Optional custom OpenRouter API key.

        Returns:
            The AI response.
        """
        # Load context from file if not present in memory
        if KaizenStat._last_context is None:
            try:
                with open(".kaizenstat_context.json", "r") as f:
                    KaizenStat._last_context = json.load(f)
            except Exception:
                pass

        context = KaizenStat._last_context
        if context is None:
            raise ValueError(
                "No dataset context found. Please run KaizenStat.analyze(df, target) "
                "or KaizenStat.auto(df, target) first."
            )

        # Load history from file if not present in memory (critical for CLI flow)
        if not KaizenStat._conversation_history:
            try:
                with open(".kaizenstat_history.json", "r") as f:
                    KaizenStat._conversation_history = json.load(f)
            except Exception:
                pass

        if not KaizenStat._conversation_history:
            return KaizenStat.ask(user_query, api_key=api_key)

        history = KaizenStat._conversation_history
        history.append({"role": "user", "content": user_query})

        system_prompt = KaizenStat._get_system_prompt(context)
        messages = [{"role": "system", "content": system_prompt}] + history

        print(f"\n🧠 Querying KaizenStat (Follow-up) for: '{user_query}'...")
        response = KaizenStat._call_openrouter_api_messages(messages, api_key=api_key)

        history.append({"role": "assistant", "content": response})
        KaizenStat._conversation_history = history

        # Persist conversation history to disk
        try:
            with open(".kaizenstat_history.json", "w") as f:
                json.dump(KaizenStat._conversation_history, f, indent=2)
        except Exception:
            pass

        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()
        console.print()
        console.print(Panel(Markdown(response), title="[bold green]💬 KAIZENSTAT RESPONSE (FOLLOW-UP)[/]", border_style="cyan"))
        return SilentStr(response)

    @staticmethod
    def improve(api_key: Optional[str] = None) -> str:
        """
        Query KaizenStat Intelligence to get the next best actions/improvement plan.

        Args:
            api_key: Optional custom OpenRouter API key.

        Returns:
            The plain-English actionable improvement plan.
        """
        return KaizenStat.ask(
            "What should I improve first and what is the actionable plan?",
            api_key=api_key
        )