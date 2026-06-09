"""Model benchmarking and training engine."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, train_test_split, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from kaizenstat.utils.helpers import (
    detect_task_type,
    get_categorical_cols,
    get_numeric_cols,
    validate_dataframe,
)

warnings.filterwarnings("ignore")
console = Console()

# Hyperparameter search spaces — keyed by model name from _get_models()
# Wider grids give progressive tuning more room to improve on the coarse pass.
_PARAM_GRIDS: Dict[str, Dict] = {
    "LogisticRegression": {
        "model__C":        [0.001, 0.01, 0.1, 1, 10, 100],
        "model__solver":   ["lbfgs", "saga"],
        "model__penalty":  ["l2"],
    },
    "RandomForest": {
        "model__n_estimators":     [100, 200, 300, 500],
        "model__max_depth":        [None, 5, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features":     ["sqrt", "log2", 0.3, 0.5],
        "model__min_samples_split":[2, 5, 10],
    },
    "GradientBoosting": {
        "model__n_estimators":  [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth":     [3, 4, 5, 7],
        "model__subsample":     [0.7, 0.8, 0.9, 1.0],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "XGBoost": {
        "model__n_estimators":     [100, 200, 300, 500],
        "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "model__max_depth":        [3, 4, 5, 7, 9],
        "model__subsample":        [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_alpha":        [0, 0.01, 0.1],
        "model__reg_lambda":       [1, 1.5, 2],
    },
    "LightGBM": {
        "model__n_estimators":  [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__num_leaves":    [15, 31, 63, 127],
        "model__subsample":     [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_samples": [10, 20, 50],
        "model__reg_alpha":    [0, 0.01, 0.1],
    },
    "Ridge": {
        "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    },
    "ExtraTreesClassifier": {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features":     ["sqrt", "log2"],
    },
    "ExtraTreesRegressor": {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    },
}


@dataclass
class ModelEntry:
    name: str
    score: float
    std: float
    train_time: float
    metric: str
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    task: str                  # classification / regression
    metric: str
    entries: List[ModelEntry]
    best_name: str
    best_score: float
    best_pipeline: Optional[Any] = None
    label_encoder: Optional[Any] = None

    def display(self) -> None:
        console.print(Panel.fit(
            f"[bold cyan]Task: {self.task.upper()}[/bold cyan]  │  "
            f"Metric: {self.metric}  │  "
            f"Best: [bold green]{self.best_name}[/bold green] ({self.best_score:.4f})",
            title="[bold]KaizenStat · Model Benchmark[/bold]",
            border_style="cyan",
        ))

        table = Table(box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Rank", justify="center", width=6)
        table.add_column("Model", style="cyan")
        table.add_column("Score", justify="right", style="bold green")
        table.add_column("± Std", justify="right", style="dim")
        table.add_column("Time(s)", justify="right", style="yellow")

        medals = ["🥇", "🥈", "🥉"]
        for i, e in enumerate(sorted(self.entries, key=lambda x: x.score, reverse=True)):
            rank = medals[i] if i < 3 else f"#{i + 1}"
            score_str = f"{e.score:.4f}" if e.error is None else "[red]ERROR[/red]"
            table.add_row(rank, e.name, score_str, f"± {e.std:.4f}", f"{e.train_time:.2f}s")
        console.print(table)
        console.print()


@dataclass
class TrainResult:
    model_name: str
    task: str
    train_score: float
    test_score: float
    metrics: Dict[str, float]
    pipeline: Any
    label_encoder: Optional[Any] = None
    feature_names: List[str] = field(default_factory=list)
    cv_score: float = 0.0
    cv_std: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)

    def display(self) -> None:
        tuned = f"  │  [bold magenta]Tuned ✓[/bold magenta]" if self.best_params else ""
        cv_line = (f"CV Score: [cyan]{self.cv_score:.4f}[/cyan] ± {self.cv_std:.4f}  │  " if self.cv_score else "")
        console.print(Panel.fit(
            f"[bold cyan]{self.model_name}[/bold cyan]  │  Task: {self.task}{tuned}\n"
            f"{cv_line}"
            f"Train: [green]{self.train_score:.4f}[/green]  │  "
            f"Test:  [bold green]{self.test_score:.4f}[/bold green]",
            title="[bold]KaizenStat · Train Result[/bold]",
            border_style="cyan",
        ))
        if self.metrics:
            for k, v in self.metrics.items():
                console.print(f"  [cyan]{k}:[/cyan] {v:.4f}")
        if self.best_params:
            console.print("  [bold magenta]Best params:[/bold magenta]")
            for k, v in self.best_params.items():
                console.print(f"    {k.replace('model__', '')}: {v}")
        console.print()


class ModelTrainer:
    """Benchmarks and trains ML models with automatic preprocessing."""

    def benchmark(
        self,
        df: pd.DataFrame,
        target: str,
        cv: int = 5,
        extra_models: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        validate_dataframe(df, target)
        X, y, task, le = self._prepare(df, target)
        metric = self._select_metric(task, y)
        models = self._get_models(task, y, extra_models=extra_models)
        preprocessor = self._build_preprocessor(X)
        n_cv = min(cv, max(2, len(X) // 50))

        entries: List[ModelEntry] = []
        best_score = -np.inf
        best_pipeline = None

        for name, model in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            with Progress(SpinnerColumn(), TextColumn(f"Training [cyan]{name}[/cyan]..."),
                          console=console, transient=True) as prog:
                prog.add_task("", total=None)
                t0 = time.time()
                try:
                    cv_obj = (
                        StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
                        if task == "classification" else
                        KFold(n_splits=n_cv, shuffle=True, random_state=42)
                    )
                    scores = cross_val_score(pipe, X, y, cv=cv_obj, scoring=metric, n_jobs=-1)
                    elapsed = time.time() - t0
                    mean_score = float(scores.mean())

                    if mean_score > best_score:
                        best_score = mean_score
                        pipe.fit(X, y)
                        best_pipeline = pipe

                    entries.append(ModelEntry(
                        name=name, score=round(mean_score, 4),
                        std=round(float(scores.std()), 4),
                        train_time=round(elapsed, 2),
                        metric=metric,
                    ))
                except Exception as exc:
                    entries.append(ModelEntry(
                        name=name, score=0.0, std=0.0,
                        train_time=round(time.time() - t0, 2),
                        metric=metric, error=str(exc)[:80],
                    ))

        best = max(entries, key=lambda e: e.score)
        result = BenchmarkResult(
            task=task, metric=metric, entries=entries,
            best_name=best.name, best_score=best.score,
            best_pipeline=best_pipeline, label_encoder=le,
        )
        result.display()
        return result

    def train_best(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        cv: int = 5,
        tune: bool = False,
        n_iter: int = 20,
        extra_models: Optional[Dict[str, Any]] = None,
    ) -> TrainResult:
        validate_dataframe(df, target)
        X, y, task, le = self._prepare(df, target)

        # Split FIRST so the benchmark only sees training data.
        # Fitting benchmark on full data then scoring on the split would cause
        # test-set contamination (X_test rows already seen during fit → inflated scores).
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                stratify=y if task == "classification" else None,
                random_state=42,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Reconstruct a training-only DataFrame to pass to benchmark
        train_df = X_train.copy()
        train_df[target] = y_train.values

        bm = self.benchmark(train_df, target, cv=cv, extra_models=extra_models)
        if bm.best_pipeline is None:
            raise RuntimeError("No model was successfully trained during benchmark.")  # pragma: no cover

        # Find the CV score for the best model entry
        best_entry = next((e for e in bm.entries if e.name == bm.best_name), None)
        cv_score = best_entry.score if best_entry else 0.0
        cv_std   = best_entry.std   if best_entry else 0.0

        pipe        = bm.best_pipeline
        best_params = {}

        # Optional hyperparameter tuning — progressive 2-stage search
        if tune and bm.best_name in _PARAM_GRIDS:
            param_grid  = _PARAM_GRIDS[bm.best_name]
            pipe, best_params, cv_score = self._progressive_tune(
                pipe, param_grid, X_train, y_train, task, cv, n_iter
            )

        # Calibrate final classification model when predict_proba is available
        if task == "classification":
            try:
                proba = pipe.predict_proba(X_train[:min(200, len(X_train))])
                confidence = float(proba.max(axis=1).mean())
                if confidence > 0.95:
                    from sklearn.metrics import accuracy_score as _acc
                    acc = float(_acc(y_train[:min(200, len(y_train))],
                                     pipe.predict(X_train[:min(200, len(X_train))])))
                    if confidence - acc > 0.10:
                        cal = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")  # pragma: no cover
                        cal.fit(X_train, y_train)  # pragma: no cover
                        pipe = cal  # pragma: no cover
                        console.print("[dim]Applied Platt calibration — model was overconfident[/dim]")  # pragma: no cover
            except Exception:  # pragma: no cover
                pass

        train_score = pipe.score(X_train, y_train)
        test_score  = pipe.score(X_test,  y_test)

        metrics = self._compute_metrics(pipe, X_test, y_test, task)
        result = TrainResult(
            model_name=bm.best_name, task=task,
            train_score=round(train_score, 4), test_score=round(test_score, 4),
            metrics=metrics, pipeline=pipe, label_encoder=le,
            feature_names=list(X.columns),
            cv_score=cv_score, cv_std=cv_std, best_params=best_params,
        )
        result.display()
        return result

    def evaluate(
        self,
        pipeline: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: Optional[str] = None,
    ) -> Dict[str, float]:
        if task is None:
            task = detect_task_type(y_test)
        metrics = self._compute_metrics(pipeline, X_test, y_test, task)
        console.print("[bold cyan]Evaluation Metrics[/bold cyan]")
        for k, v in metrics.items():
            console.print(f"  {k}: [bold]{v:.4f}[/bold]")
        return metrics

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare(self, df, target) -> Tuple[pd.DataFrame, pd.Series, str, Optional[LabelEncoder]]:
        df = df.copy()
        y = df[target].dropna()
        df = df.loc[y.index]
        X = df.drop(columns=[target])
        # Only coerce object columns that look numeric (e.g. "3.14") — leave string cols
        # like Sex/Embarked as object so the OHE preprocessor can encode them properly.
        def _try_numeric(col):
            if col.dtype != object:
                return col
            converted = pd.to_numeric(col, errors="coerce")
            # Keep conversion only if ≥80% of non-null values converted successfully
            non_null = col.notna().sum()
            if non_null > 0 and converted.notna().sum() / non_null >= 0.8:
                return converted
            return col
        X = X.apply(_try_numeric)

        task = detect_task_type(y)
        le = None
        if task == "classification" and y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index, name=target)

        return X, y, task, le

    @staticmethod
    def _make_ohe() -> OneHotEncoder:
        """Build OneHotEncoder compatible with sklearn 1.1+ and 1.2+."""
        try:
            # sparse_output added in sklearn 1.2
            return OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False)
        except TypeError:  # pragma: no cover
            try:
                # max_categories added in sklearn 1.1; sparse renamed in 1.2
                return OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse=False)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore", sparse=False)

    @staticmethod
    def _select_metric(task: str, y: pd.Series) -> str:
        """Use F1-weighted for imbalanced classification; accuracy for balanced; R2 for regression."""
        if task != "classification":
            return "r2"
        counts = y.value_counts(normalize=True)
        if len(counts) >= 2 and counts.iloc[-1] < 0.20:
            return "f1_weighted"
        return "accuracy"

    @staticmethod
    def _analyze_data(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Build a data intelligence profile used for smart model and pipeline selection."""
        counts = y.value_counts(normalize=True)
        return {
            "n_rows":        X.shape[0],
            "n_cols":        X.shape[1],
            "num_cols":      X.select_dtypes(include=["number"]).shape[1],
            "cat_cols":      X.select_dtypes(include=["object", "category"]).shape[1],
            "imbalance":     float(counts.min()) if len(counts) > 1 else 1.0,
            "missing_ratio": float(X.isna().mean().mean()),
            "high_dim":      X.shape[1] > 50,
            "sparse":        bool(
                (X.select_dtypes(include="number") == 0).mean().mean() > 0.5
            ),
        }

    def _get_models(
        self,
        task: str,
        y: pd.Series,
        extra_models: Optional[Dict[str, Any]] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if profile is None:
            profile = {}

        imbalanced = profile.get("imbalance", 1.0) < 0.15
        if not imbalanced and task == "classification":
            counts = y.value_counts(normalize=True)
            imbalanced = len(counts) > 1 and counts.iloc[-1] < 0.15

        high_dim = profile.get("high_dim", False)

        models: Dict[str, Any] = {}
        if task == "classification":
            cw = "balanced" if imbalanced else None
            models["LogisticRegression"] = LogisticRegression(
                max_iter=1000, class_weight=cw, random_state=42
            )
            models["RandomForest"] = RandomForestClassifier(
                n_estimators=100, class_weight=cw, random_state=42
            )
            if not high_dim:
                models["GradientBoosting"] = GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                )
        else:
            models["Ridge"] = Ridge()
            models["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=42)
            if not high_dim:
                models["GradientBoosting"] = GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                )

        # ExtraTrees — fast, diverse, complements Random Forest well in stacking
        if task == "classification":
            models["ExtraTreesClassifier"] = ExtraTreesClassifier(
                n_estimators=100, class_weight=cw if task == "classification" else None,
                random_state=42,
            )
        else:
            models["ExtraTreesRegressor"] = ExtraTreesRegressor(
                n_estimators=100, random_state=42,
            )

        # Boosting libraries — always preferred on high-dim or imbalanced data
        try:
            import xgboost as xgb
            if task == "classification":
                models["XGBoost"] = xgb.XGBClassifier(
                    tree_method="hist", random_state=42, eval_metric="logloss",
                    n_estimators=200, learning_rate=0.1,
                )
            else:
                models["XGBoost"] = xgb.XGBRegressor(
                    tree_method="hist", random_state=42,
                    n_estimators=200, learning_rate=0.1,
                )
        except ImportError:  # pragma: no cover
            pass

        try:  # pragma: no cover
            import lightgbm as lgb  # pragma: no cover
            if task == "classification":  # pragma: no cover
                models["LightGBM"] = lgb.LGBMClassifier(  # pragma: no cover
                    random_state=42, verbose=-1,  # pragma: no cover
                    n_estimators=200, learning_rate=0.1, num_leaves=31,  # pragma: no cover
                )  # pragma: no cover
            else:  # pragma: no cover
                models["LightGBM"] = lgb.LGBMRegressor(  # pragma: no cover
                    random_state=42, verbose=-1,  # pragma: no cover
                    n_estimators=200, learning_rate=0.1,  # pragma: no cover
                )  # pragma: no cover
        except ImportError:  # pragma: no cover
            pass

        if extra_models:
            models.update(extra_models)

        return models

    def _build_preprocessor(self, X: pd.DataFrame, high_dim: bool = False) -> ColumnTransformer:
        num = get_numeric_cols(X)
        cat = get_categorical_cols(X)
        transformers = []
        if num:
            steps: list = [("scaler", StandardScaler())]
            if high_dim and len(num) > 50:
                k = min(50, len(num))
                steps.append(("selector", SelectKBest(f_regression, k=k)))
            from sklearn.pipeline import Pipeline as _P
            transformers.append(("num", _P(steps), num))
        if cat:
            transformers.append(("cat", self._make_ohe(), cat))
        if not transformers:
            raise ValueError("No usable feature columns found.")
        return ColumnTransformer(transformers, remainder="drop")

    def _build_ensemble(
        self, named_pipelines: List[Tuple[str, Any]], task: str
    ) -> Any:
        """Soft-voting ensemble — kept for backward compat; _build_stacking_ensemble preferred."""
        if task == "classification":
            return VotingClassifier(estimators=named_pipelines, voting="soft")
        return VotingRegressor(estimators=named_pipelines)

    def _build_stacking_ensemble(
        self, named_pipelines: List[Tuple[str, Any]], task: str
    ) -> Any:
        """
        Stacking ensemble with a meta-learner (top Kaggle / competition approach).

        Base models produce out-of-fold predictions via cv=3; the meta-learner
        (LogReg / Ridge) learns the optimal combination — outperforms soft voting
        because it can assign negative weights to weak base models.
        """
        if task == "classification":
            meta = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            return StackingClassifier(
                estimators=named_pipelines,
                final_estimator=meta,
                cv=3,
                passthrough=False,
            )
        return StackingRegressor(
            estimators=named_pipelines,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
        )

    def _narrow_param_grid(
        self, param_grid: Dict[str, list], best_params: Dict[str, Any]
    ) -> Dict[str, list]:
        """Build a refined grid ±1 neighbour around each best value for stage-2 search."""
        narrow: Dict[str, list] = {}
        for key, candidates in param_grid.items():
            if key in best_params and isinstance(candidates, list):
                best_val = best_params[key]
                if best_val in candidates:
                    idx = candidates.index(best_val)
                    lo  = max(0, idx - 1)
                    hi  = min(len(candidates), idx + 2)
                    narrow[key] = candidates[lo:hi] or [best_val]
                else:
                    narrow[key] = candidates  # pragma: no cover
            else:
                narrow[key] = candidates  # pragma: no cover
        return narrow

    def _progressive_tune(
        self,
        pipe: Any,
        param_grid: Dict[str, list],
        X_train: Any,
        y_train: Any,
        task: str,
        cv: int,
        n_iter: int,
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        2-stage progressive tuning: coarse random search → refine around best params.

        Stage 1 uses half the iterations on the full grid.
        Stage 2 uses the remaining iterations on a narrowed grid centred on the
        stage-1 winner, reliably beating a single-stage search of the same budget.
        """
        metric = self._select_metric(task, y_train)
        cv_obj = (
            StratifiedKFold(n_splits=min(cv, 5), shuffle=True, random_state=42)
            if task == "classification" else
            KFold(n_splits=min(cv, 5), shuffle=True, random_state=42)
        )

        coarse_iter = max(5, n_iter // 2)
        console.print(f"[bold magenta]Progressive tuning — Stage 1/2: coarse search (n_iter={coarse_iter})…[/bold magenta]")
        with Progress(SpinnerColumn(), TextColumn("Coarse hyperparameter search…"),
                      console=console, transient=True) as prog:
            prog.add_task("", total=None)
            coarse = RandomizedSearchCV(
                pipe, param_grid, n_iter=coarse_iter,
                cv=cv_obj, scoring=metric,
                random_state=42, n_jobs=-1, refit=True,
            )
            coarse.fit(X_train, y_train)

        fine_grid  = self._narrow_param_grid(param_grid, coarse.best_params_)
        fine_iter  = max(5, n_iter - coarse_iter)
        console.print(f"[bold magenta]Progressive tuning — Stage 2/2: refining (n_iter={fine_iter})…[/bold magenta]")
        with Progress(SpinnerColumn(), TextColumn("Fine hyperparameter search…"),
                      console=console, transient=True) as prog:
            prog.add_task("", total=None)
            fine = RandomizedSearchCV(
                pipe, fine_grid, n_iter=fine_iter,
                cv=cv_obj, scoring=metric,
                random_state=0, n_jobs=-1, refit=True,
            )
            fine.fit(X_train, y_train)

        if fine.best_score_ >= coarse.best_score_:
            console.print(
                f"[green]✓ Progressive tuning — coarse: {coarse.best_score_:.4f} → "
                f"fine: {fine.best_score_:.4f} (+{fine.best_score_ - coarse.best_score_:.4f})[/green]"
            )
            return fine.best_estimator_, fine.best_params_, round(float(fine.best_score_), 4)

        console.print(f"[green]✓ Progressive tuning — best: {coarse.best_score_:.4f}[/green]")
        return coarse.best_estimator_, coarse.best_params_, round(float(coarse.best_score_), 4)

    def train_auto(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        cv: int = 3,
        tune: bool = False,
        n_iter: int = 10,
        ensemble: bool = True,
        extra_models: Optional[Dict[str, Any]] = None,
    ) -> TrainResult:
        """
        Full AutoML pipeline:
          1. Build data intelligence profile (imbalance, dimensionality, sparsity)
          2. Profile-aware model selection
          3. Optional feature selection (high-dim data)
          4. Optional hyperparameter tuning on the best model
          5. Soft-voting ensemble of the top 3 models (optional)

        Args:
            ensemble: If True, return a soft-voting ensemble of the top 3 models
                      instead of just the single best model.
        """
        validate_dataframe(df, target)
        X, y, task, le = self._prepare(df, target)

        # 1. Data intelligence profile
        profile = self._analyze_data(X, y)
        console.print(Panel.fit(
            f"[bold cyan]Data Profile[/bold cyan]\n"
            f"  Rows: {profile['n_rows']:,}  │  Numeric: {profile['num_cols']}  │  "
            f"Categorical: {profile['cat_cols']}\n"
            f"  Imbalance: {profile['imbalance']:.0%}  │  "
            f"Missing: {profile['missing_ratio']:.1%}  │  "
            f"High-dim: {profile['high_dim']}",
            title="[bold]KaizenStat · AutoML — Data Intelligence[/bold]",
            border_style="cyan",
        ))

        # 2. Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                stratify=y if task == "classification" else None,
                random_state=42,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        train_df = X_train.copy()
        train_df[target] = y_train.values

        # 3. Benchmark with profile-aware model set
        bm = self.benchmark(train_df, target, cv=cv, extra_models=extra_models)
        if bm.best_pipeline is None:
            raise RuntimeError("No model was successfully trained during benchmark.")  # pragma: no cover

        best_entry = next((e for e in bm.entries if e.name == bm.best_name), None)
        cv_score = best_entry.score if best_entry else 0.0
        cv_std   = best_entry.std   if best_entry else 0.0

        pipe        = bm.best_pipeline
        best_params = {}

        # 4. Optional progressive 2-stage tuning on best model
        if tune and bm.best_name in _PARAM_GRIDS:
            param_grid = _PARAM_GRIDS[bm.best_name]
            pipe, best_params, cv_score = self._progressive_tune(
                pipe, param_grid, X_train, y_train, task, cv, n_iter
            )

        # 5. Stacking ensemble from top-3 diverse models (outperforms soft voting)
        model_name = bm.best_name
        if ensemble:
            sorted_entries = sorted(
                [e for e in bm.entries if e.error is None],
                key=lambda e: e.score, reverse=True,
            )
            top_names = [e.name for e in sorted_entries[:3]]
            if len(top_names) >= 2:
                console.print(
                    f"[bold cyan]Building stacking ensemble from top {len(top_names)} models: "
                    f"{top_names}[/bold cyan]\n"
                    f"[dim]Meta-learner: LogisticRegression/Ridge with out-of-fold predictions[/dim]"
                )
                preprocessor = self._build_preprocessor(X_train, high_dim=profile["high_dim"])
                models_map   = self._get_models(task, y_train, extra_models=extra_models, profile=profile)
                named_pipes: List[Tuple[str, Any]] = []
                for name in top_names:
                    if name not in models_map:
                        continue  # pragma: no cover
                    p = Pipeline([("prep", preprocessor), ("model", models_map[name])])
                    p.fit(X_train, y_train)
                    named_pipes.append((name, p))
                if len(named_pipes) >= 2:
                    try:
                        ens = self._build_stacking_ensemble(named_pipes, task)
                        ens.fit(X_train, y_train)
                        pipe       = ens
                        model_name = f"Stack({'+'.join(top_names)})"
                        console.print(f"[green]✓ Stacking ensemble trained[/green]")
                    except Exception as exc:
                        console.print(f"[yellow]Stacking failed ({exc}) — falling back to soft voting[/yellow]")
                        ens = self._build_ensemble(named_pipes, task)
                        ens.fit(X_train, y_train)
                        pipe       = ens
                        model_name = f"Ensemble({'+'.join(top_names)})"

        # Calibrate final classification model if overconfident
        if task == "classification":
            try:
                proba = pipe.predict_proba(X_train[:min(200, len(X_train))])
                confidence = float(proba.max(axis=1).mean())
                if confidence > 0.95:  # pragma: no cover
                    from sklearn.metrics import accuracy_score as _acc  # pragma: no cover
                    acc = float(_acc(y_train[:min(200, len(y_train))],  # pragma: no cover
                                     pipe.predict(X_train[:min(200, len(X_train))])))  # pragma: no cover
                    if confidence - acc > 0.10:  # pragma: no cover
                        cal = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")  # pragma: no cover
                        cal.fit(X_train, y_train)  # pragma: no cover
                        pipe = cal  # pragma: no cover
                        console.print("[dim]Applied Platt calibration — ensemble was overconfident[/dim]")  # pragma: no cover
            except Exception:  # pragma: no cover
                pass

        train_score = pipe.score(X_train, y_train)
        test_score  = pipe.score(X_test,  y_test)
        metrics     = self._compute_metrics(pipe, X_test, y_test, task)

        result = TrainResult(
            model_name=model_name, task=task,
            train_score=round(train_score, 4), test_score=round(test_score, 4),
            metrics=metrics, pipeline=pipe, label_encoder=le,
            feature_names=list(X.columns),
            cv_score=cv_score, cv_std=cv_std, best_params=best_params,
        )
        result.display()
        return result

    def _compute_metrics(self, pipe, X_test, y_test, task) -> Dict[str, float]:
        y_pred = pipe.predict(X_test)
        if task == "classification":
            metrics: Dict[str, float] = {
                "accuracy": accuracy_score(y_test, y_pred),
            }
            try:
                n_classes = len(np.unique(y_test))
                avg = "binary" if n_classes == 2 else "weighted"
                metrics["f1"] = f1_score(y_test, y_pred, average=avg, zero_division=0)
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    if n_classes == 2:
                        metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                    else:
                        metrics["roc_auc"] = roc_auc_score(
                            y_test, proba, multi_class="ovr", average="weighted"
                        )
            except Exception:
                pass
        else:
            metrics = {
                "r2": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
            }
        return {k: round(v, 4) for k, v in metrics.items()}


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_trainer = ModelTrainer()


def benchmark(
    df: pd.DataFrame, target: str, cv: int = 5,
    extra_models: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    return _trainer.benchmark(df, target, cv=cv, extra_models=extra_models)


def train_best(
    df: pd.DataFrame, target: str,
    test_size: float = 0.2, cv: int = 5,
    tune: bool = False, n_iter: int = 20,
    extra_models: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    return _trainer.train_best(
        df, target, test_size=test_size, cv=cv,
        tune=tune, n_iter=n_iter, extra_models=extra_models,
    )


def evaluate(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: Optional[str] = None,
) -> Dict[str, float]:
    return _trainer.evaluate(pipeline, X_test, y_test, task=task)


def train_auto(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    cv: int = 3,
    tune: bool = False,
    n_iter: int = 10,
    ensemble: bool = True,
    extra_models: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    return _trainer.train_auto(
        df, target, test_size=test_size, cv=cv,
        tune=tune, n_iter=n_iter, ensemble=ensemble,
        extra_models=extra_models,
    )
