"""Text model training engine — NLP counterpart of trainer.py.

Size-adaptive pipeline selection:
    <  5k rows : TF-IDF + LogisticRegression   (fast, strong baseline)
    5k–50k     : TF-IDF + LinearSVC            (best linear text classifier)
    > 50k      : TF-IDF (hashing-style, capped) + SGDClassifier (scales out-of-core)

Reuses TrainResult so DataDoctor returns the same structured output as tabular.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from kaizenstat.model.trainer import TrainResult
from kaizenstat.utils.helpers import (
    detect_task_type, dominant_text_column, validate_dataframe,
)

warnings.filterwarnings("ignore")
console = Console()

# Adaptive hyperparameter search spaces (joint vectoriser + model)
_TEXT_PARAM_GRIDS: Dict[str, Dict] = {
    "TFIDF+LogReg": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df":      [1, 2, 5],
        "tfidf__max_df":      [0.9, 0.95, 1.0],
        "tfidf__sublinear_tf":[True, False],
        "model__C":           [0.1, 1, 10],
    },
    "TFIDF+LinearSVC": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df":      [1, 2, 5],
        "tfidf__sublinear_tf":[True, False],
        "model__estimator__C":[0.1, 1, 10],
    },
    "TFIDF+SGD": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df":      [2, 5],
        "model__alpha":       [1e-5, 1e-4, 1e-3],
    },
}


class TextModelTrainer:
    """Trains size-adaptive TF-IDF text classifiers / regressors."""

    def train_best(
        self,
        df: pd.DataFrame,
        target: str,
        text_col: Optional[str] = None,
        test_size: float = 0.2,
        cv: int = 5,
        tune: bool = False,
        n_iter: int = 15,
    ) -> TrainResult:
        validate_dataframe(df, target)
        text_col = text_col or dominant_text_column(df, exclude=[target])
        if text_col is None:
            raise ValueError("No dominant text column found for text training.")

        X_text, y, task, le = self._prepare(df, target, text_col)
        n = len(X_text)

        pipe_name, pipe = self._select_pipeline(n, task)
        metric = self._select_metric(task, y)

        console.print(Panel.fit(
            f"[bold cyan]Text column:[/bold cyan] '{text_col}'  │  "
            f"Rows: {n:,}  │  Task: {task}\n"
            f"[bold cyan]Selected pipeline:[/bold cyan] {pipe_name}  │  Metric: {metric}",
            title="[bold]KaizenStat · Text AutoML[/bold]",
            border_style="cyan",
        ))

        # Leakage-free split first
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_text, y, test_size=test_size,
                stratify=y if task == "classification" else None,
                random_state=42,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_text, y, test_size=test_size, random_state=42
            )

        n_cv = min(cv, max(2, len(X_tr) // 50))
        cv_obj = (
            StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
            if task == "classification" else
            KFold(n_splits=n_cv, shuffle=True, random_state=42)
        )

        # Benchmark multiple text pipelines and pick the best one
        if task == "classification":
            pipe_name, pipe, cv_score = self._benchmark_text_pipelines(
                X_tr, y_tr, task, cv_obj, metric
            )
            console.print(
                f"[green]✓ Best text pipeline: [bold]{pipe_name}[/bold] "
                f"(CV {cv_score:.4f})[/green]"
            )
        else:
            with Progress(SpinnerColumn(), TextColumn(f"Cross-validating {pipe_name}…"),
                          console=console, transient=True) as prog:
                prog.add_task("", total=None)
                scores = cross_val_score(pipe, X_tr, y_tr, cv=cv_obj, scoring=metric, n_jobs=-1)
                cv_score = round(float(scores.mean()), 4)

        # Try sentence embeddings if available — upgrade if they beat TF-IDF
        emb_result = self._try_sentence_embeddings(X_tr, y_tr, task, cv_obj, metric)
        if emb_result is not None:  # pragma: no cover
            emb_name, emb_score, emb_pipe = emb_result  # pragma: no cover
            if emb_score > cv_score + 0.01:  # pragma: no cover
                console.print(  # pragma: no cover
                    f"[bold green]✓ Sentence embeddings ({emb_name}) beat TF-IDF: "  # pragma: no cover
                    f"{emb_score:.4f} vs {cv_score:.4f} → upgrading[/bold green]"  # pragma: no cover
                )  # pragma: no cover
                pipe_name = emb_name  # pragma: no cover
                cv_score  = emb_score  # pragma: no cover
                pipe      = emb_pipe  # pragma: no cover

        cv_std: float = 0.0
        best_params: Dict[str, Any] = {}

        if tune and pipe_name in _TEXT_PARAM_GRIDS:
            console.print(f"[bold magenta]Tuning {pipe_name} (n_iter={n_iter})…[/bold magenta]")
            with Progress(SpinnerColumn(), TextColumn("Searching text hyperparameters…"),
                          console=console, transient=True) as prog:
                prog.add_task("", total=None)
                search = RandomizedSearchCV(
                    pipe, _TEXT_PARAM_GRIDS[pipe_name],
                    n_iter=n_iter, cv=cv_obj, scoring=metric,
                    random_state=42, n_jobs=-1, refit=True,
                )
                search.fit(X_tr, y_tr)
            pipe        = search.best_estimator_
            best_params = search.best_params_
            cv_score    = round(float(search.best_score_), 4)
            console.print(f"[green]✓ Tuning complete — best CV: {cv_score:.4f}[/green]")
        else:
            pipe.fit(X_tr, y_tr)

        train_score = pipe.score(X_tr, y_tr)
        test_score  = pipe.score(X_te, y_te)
        metrics     = self._compute_metrics(pipe, X_te, y_te, task)

        result = TrainResult(
            model_name=pipe_name, task=task,
            train_score=round(train_score, 4), test_score=round(test_score, 4),
            metrics=metrics, pipeline=pipe, label_encoder=le,
            feature_names=[text_col],
            cv_score=cv_score, cv_std=cv_std, best_params=best_params,
        )
        result.display()
        return result

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _prepare(self, df, target, text_col) -> Tuple[pd.Series, pd.Series, str, Optional[LabelEncoder]]:
        df = df.copy()
        y = df[target].dropna()
        df = df.loc[y.index]
        X_text = df[text_col].fillna("").astype(str)

        task = detect_task_type(y)
        le = None
        if task == "classification" and y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index, name=target)
        return X_text, y, task, le

    def _benchmark_text_pipelines(
        self,
        X_tr: Any,
        y_tr: Any,
        task: str,  # noqa: ARG002 — reserved for future regression support
        cv_obj: Any,
        metric: str,
    ) -> Tuple[str, Any, float]:
        """
        Benchmark multiple text pipeline variants and return the best one.

        Candidates:
          1. TF-IDF word n-grams + LogReg  (fast baseline)
          2. TF-IDF char n-grams + LogReg  (robust for noisy / short text)
          3. TF-IDF word n-grams + LinearSVC (calibrated, stronger on larger sets)

        Returns (best_name, best_pipeline, best_cv_score).
        """
        n = len(X_tr)

        def _logreg():
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

        candidates: Dict[str, Any] = {}

        # Word n-gram baseline
        vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                sublinear_tf=True, strip_accents="unicode")
        candidates["TFIDF+LogReg"] = Pipeline([("tfidf", vec_w), ("model", _logreg())])

        # Char n-gram — robust to typos, morphology, short texts
        vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                min_df=2, max_df=0.95, sublinear_tf=True,
                                max_features=100_000)
        candidates["TFIDF_char+LogReg"] = Pipeline([("tfidf", vec_c), ("model", _logreg())])

        # LinearSVC (calibrated) — usually best for medium+ datasets
        if n >= 500:
            vec_s = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                    sublinear_tf=True, strip_accents="unicode")
            svc   = LinearSVC(class_weight="balanced", random_state=42)
            candidates["TFIDF+LinearSVC"] = Pipeline([
                ("tfidf", vec_s),
                ("model", CalibratedClassifierCV(svc, cv=3)),
            ])

        best_name  = "TFIDF+LogReg"
        best_pipe  = candidates["TFIDF+LogReg"]
        best_score = -1.0

        table_rows = []
        for name, pipe in candidates.items():
            try:
                with Progress(SpinnerColumn(), TextColumn(f"Benchmarking [cyan]{name}[/cyan]…"),
                              console=console, transient=True) as prog:
                    prog.add_task("", total=None)
                    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv_obj,
                                             scoring=metric, n_jobs=-1)
                mean_s = round(float(scores.mean()), 4)
                table_rows.append((name, mean_s, round(float(scores.std()), 4)))
                if mean_s > best_score:
                    best_score = mean_s
                    best_name  = name
                    best_pipe  = pipe
            except Exception:
                pass

        from rich import box as _box
        from rich.table import Table as _Table
        tbl = _Table(box=_box.SIMPLE, header_style="bold magenta",
                     title="Text Pipeline Benchmark")
        tbl.add_column("Pipeline", style="cyan")
        tbl.add_column("CV Score", justify="right")
        tbl.add_column("± Std", justify="right", style="dim")
        for row_name, s, sd in sorted(table_rows, key=lambda r: r[1], reverse=True):
            mark = " 🏆" if row_name == best_name else ""
            tbl.add_row(row_name + mark, f"{s:.4f}", f"± {sd:.4f}")
        console.print(tbl)

        best_pipe.fit(X_tr, y_tr)
        return best_name, best_pipe, best_score

    def _try_sentence_embeddings(
        self,
        X_tr: Any,
        y_tr: Any,
        task: str,
        cv_obj: Any,
        metric: str,
    ) -> Optional[Tuple[str, float, Any]]:
        """
        Try sentence-transformers embeddings (all-MiniLM-L6-v2) + LogReg.

        Returns (name, cv_score, fitted_pipeline) if available and competitive,
        or None if the package is not installed.

        Requires: pip install "kaizenstat[nlp]"
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None

        try:  # pragma: no cover
            console.print("[dim]Trying sentence embeddings (all-MiniLM-L6-v2)…[/dim]")
            encoder = SentenceTransformer("all-MiniLM-L6-v2")

            X_list = X_tr.tolist() if hasattr(X_tr, "tolist") else list(X_tr)
            X_emb  = encoder.encode(X_list, batch_size=64, show_progress_bar=False)

            clf = (LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                   if task == "classification" else Ridge())
            scores = cross_val_score(clf, X_emb, y_tr, cv=cv_obj,
                                     scoring=metric, n_jobs=-1)
            cv_score = round(float(scores.mean()), 4)

            # Return a thin wrapper that behaves like a sklearn pipeline
            class _EmbeddingPipeline:
                def __init__(self, enc, estimator):
                    self._enc  = enc
                    self._clf  = estimator

                def _embed(self, X):
                    return self._enc.encode(
                        X.tolist() if hasattr(X, "tolist") else list(X),
                        batch_size=64, show_progress_bar=False,
                    )

                def fit(self, X, y):
                    self._clf.fit(self._embed(X), y)
                    return self

                def predict(self, X):
                    return self._clf.predict(self._embed(X))

                def predict_proba(self, X):
                    if hasattr(self._clf, "predict_proba"):
                        return self._clf.predict_proba(self._embed(X))
                    raise AttributeError("Estimator has no predict_proba")

                def score(self, X, y):
                    return float((np.asarray(self.predict(X)) == np.asarray(y)).mean())

            emb_pipe = _EmbeddingPipeline(encoder, clf)
            emb_pipe.fit(X_tr, y_tr)
            return "Embeddings+LogReg", cv_score, emb_pipe

        except Exception:  # pragma: no cover
            return None

    def _select_pipeline(self, n_rows: int, task: str) -> Tuple[str, Pipeline]:
        if task != "classification":
            # Regression on text → TF-IDF + Ridge (stable, fast)
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                  sublinear_tf=True, strip_accents="unicode")
            return "TFIDF+Ridge", Pipeline([("tfidf", vec), ("model", Ridge())])

        if n_rows < 5000:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                  sublinear_tf=True, strip_accents="unicode")
            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            return "TFIDF+LogReg", Pipeline([("tfidf", vec), ("model", model)])

        if n_rows <= 50000:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                  sublinear_tf=True, strip_accents="unicode")
            # CalibratedClassifierCV wraps LinearSVC to give predict_proba (needed for trust layer)
            svc = LinearSVC(class_weight="balanced", random_state=42)
            model = CalibratedClassifierCV(svc, cv=3)
            return "TFIDF+LinearSVC", Pipeline([("tfidf", vec), ("model", model)])

        # Large: capped vocabulary + SGD (log loss → has predict_proba)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                              max_features=200_000, sublinear_tf=True, strip_accents="unicode")
        model = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42)
        return "TFIDF+SGD", Pipeline([("tfidf", vec), ("model", model)])

    @staticmethod
    def _select_metric(task: str, y: pd.Series) -> str:
        if task != "classification":
            return "r2"
        counts = y.value_counts(normalize=True)
        if len(counts) >= 2 and counts.iloc[-1] < 0.20:
            return "f1_weighted"
        return "accuracy"

    def _compute_metrics(self, pipe, X_te, y_te, task) -> Dict[str, float]:
        y_pred = pipe.predict(X_te)
        if task == "classification":
            metrics: Dict[str, float] = {"accuracy": accuracy_score(y_te, y_pred)}
            try:
                n_classes = len(np.unique(y_te))
                avg = "binary" if n_classes == 2 else "weighted"
                metrics["f1"] = f1_score(y_te, y_pred, average=avg, zero_division=0)
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_te)
                    if n_classes == 2:
                        metrics["roc_auc"] = roc_auc_score(y_te, proba[:, 1])
                    else:
                        metrics["roc_auc"] = roc_auc_score(
                            y_te, proba, multi_class="ovr", average="weighted"
                        )
            except Exception:
                pass
        else:
            metrics = {
                "r2":  r2_score(y_te, y_pred),
                "mae": mean_absolute_error(y_te, y_pred),
            }
        return {k: round(float(v), 4) for k, v in metrics.items()}


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_trainer = TextModelTrainer()


def train_best(
    df: pd.DataFrame, target: str, text_col: Optional[str] = None,
    test_size: float = 0.2, cv: int = 5, tune: bool = False, n_iter: int = 15,
) -> TrainResult:
    return _trainer.train_best(
        df, target, text_col=text_col, test_size=test_size,
        cv=cv, tune=tune, n_iter=n_iter,
    )
