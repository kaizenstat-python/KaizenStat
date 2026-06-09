"""Reliability & Trust Layer — production-readiness scoring.

Answers the question tabular metrics cannot: *can I trust this model in production?*

Components:
  • Confidence analysis   — predict_proba distribution quality
  • Prediction uncertainty — fraction of low-confidence / ambiguous predictions
  • Robustness score      — agreement under small input perturbations
  • Failure-case slicing  — which subgroups / confidence bands fail
  • Trust score (0–100)   — combined production-readiness verdict

Works for both tabular pipelines (DataFrame input) and text pipelines
(Series-of-strings input) — perturbation adapts to the input type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score

from kaizenstat.utils.helpers import detect_task_type

console = Console()


@dataclass
class TrustReport:
    trust_score: int                       # 0–100 production readiness
    grade: str                             # production-ready / needs review / not ready
    confidence_mean: float
    confidence_std: float
    uncertain_fraction: float              # share of low-confidence predictions
    robustness_score: float                # 0–1 agreement under perturbation
    calibration_gap: float                 # |confidence − accuracy| (lower is better)
    failure_slices: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Alias for trust_score — use either trust.score or trust.trust_score."""
        return self.trust_score

    def display(self) -> None:
        color = ("green" if self.trust_score >= 80 else
                 "yellow" if self.trust_score >= 60 else "red")
        console.print(Panel.fit(
            f"[bold {color}]Trust Score: {self.trust_score}/100 — {self.grade}[/bold {color}]\n"
            f"Confidence: {self.confidence_mean:.2f} ± {self.confidence_std:.2f}  │  "
            f"Uncertain: {self.uncertain_fraction:.0%}\n"
            f"Robustness: {self.robustness_score:.2f}  │  "
            f"Calibration gap: {self.calibration_gap:.3f}",
            title="[bold]KaizenStat · Reliability & Trust[/bold]",
            border_style=color,
        ))
        if self.failure_slices:
            table = Table(box=box.SIMPLE, header_style="bold magenta", title="Where it fails")
            table.add_column("Failure Slice", style="cyan")
            for s in self.failure_slices:
                table.add_row(s)
            console.print(table)
        if self.notes:
            console.print("[bold cyan]Notes:[/bold cyan]")
            for n in self.notes:
                console.print(f"  • {n}")
        console.print()


class TrustAnalyzer:
    """Computes confidence, robustness, calibration, and a 0–100 trust score."""

    def analyze(
        self,
        model: Any,
        X_test: Any,                       # DataFrame (tabular) or Series[str] (text)
        y_test: pd.Series,
        task: Optional[str] = None,
        low_conf_threshold: float = 0.60,
        n_perturb: int = 3,
    ) -> TrustReport:
        task = task or detect_task_type(y_test)
        notes: List[str] = []

        if task != "classification":
            return self._regression_trust(model, X_test, y_test, notes)

        # --- Confidence analysis ---
        proba = self._get_proba(model, X_test)
        if proba is not None:
            top_conf = proba.max(axis=1)
            conf_mean = float(top_conf.mean())
            conf_std  = float(top_conf.std())
            uncertain = float((top_conf < low_conf_threshold).mean())
        else:
            conf_mean, conf_std, uncertain = 0.5, 0.0, 1.0
            top_conf = None
            notes.append("Model has no predict_proba — confidence estimated as neutral (0.5)")

        # --- Calibration gap: |mean confidence − accuracy| ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        calibration_gap = abs(conf_mean - acc) if proba is not None else 1.0 - acc

        # --- Robustness under perturbation ---
        robustness = self._robustness(model, X_test, y_pred, n_perturb)

        # --- Failure-case slicing ---
        slices = self._failure_slices(X_test, y_test, y_pred, top_conf, low_conf_threshold)

        # --- Trust score ---
        trust = self._trust_score(acc, conf_mean, uncertain, robustness, calibration_gap)
        grade = ("production-ready" if trust >= 80 else
                 "needs review"     if trust >= 60 else "not ready")

        if calibration_gap > 0.15:
            notes.append(f"Over/under-confident: confidence and accuracy differ by {calibration_gap:.0%}")
        if robustness < 0.85:
            notes.append(f"Fragile: only {robustness:.0%} of predictions survive small input perturbations")
        if uncertain > 0.30:
            notes.append(f"{uncertain:.0%} of predictions are low-confidence (<{low_conf_threshold:.0%})")

        report = TrustReport(
            trust_score=trust, grade=grade,
            confidence_mean=round(conf_mean, 4), confidence_std=round(conf_std, 4),
            uncertain_fraction=round(uncertain, 4),
            robustness_score=round(robustness, 4),
            calibration_gap=round(calibration_gap, 4),
            failure_slices=slices, notes=notes,
        )
        report.display()
        return report

    # ------------------------------------------------------------------ #
    # Confidence / proba                                                  #
    # ------------------------------------------------------------------ #

    def _get_proba(self, model, X) -> Optional[np.ndarray]:
        if hasattr(model, "predict_proba"):
            try:
                return np.asarray(model.predict_proba(X))
            except Exception:
                return None
        if hasattr(model, "decision_function"):
            try:
                d = np.asarray(model.decision_function(X))
                if d.ndim == 1:                      # binary → sigmoid
                    p1 = 1 / (1 + np.exp(-d))
                    return np.column_stack([1 - p1, p1])
                e = np.exp(d - d.max(axis=1, keepdims=True))   # multiclass → softmax
                return e / e.sum(axis=1, keepdims=True)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------ #
    # Robustness                                                          #
    # ------------------------------------------------------------------ #

    def _robustness(self, model, X, base_pred, n_perturb: int) -> float:
        agreements = []
        for seed in range(n_perturb):
            Xp = self._perturb(X, seed)
            if Xp is None:
                return 1.0
            try:
                p = model.predict(Xp)
                agreements.append(np.mean(np.asarray(p) == np.asarray(base_pred)))
            except Exception:
                continue
        return float(np.mean(agreements)) if agreements else 1.0

    def _perturb(self, X, seed: int):
        rng = np.random.RandomState(seed)
        # Text: word dropout
        if isinstance(X, pd.Series) and X.dtype == object:
            def drop_words(t: str) -> str:
                words = str(t).split()
                if len(words) <= 3:
                    return t
                keep = [w for w in words if rng.rand() > 0.10]
                return " ".join(keep) if keep else t
            return X.apply(drop_words)
        # Tabular: Gaussian jitter on numeric columns (2% of std)
        if isinstance(X, pd.DataFrame):
            Xp = X.copy()
            num = Xp.select_dtypes(include=[np.number]).columns
            for c in num:
                std = Xp[c].std()
                if std and std > 0:
                    Xp[c] = Xp[c] + rng.normal(0, 0.02 * std, size=len(Xp))
            return Xp
        return None

    # ------------------------------------------------------------------ #
    # Failure slicing                                                     #
    # ------------------------------------------------------------------ #

    def _failure_slices(self, X, y_test, y_pred, top_conf, thr) -> List[str]:
        slices: List[str] = []
        correct = np.asarray(y_test) == np.asarray(y_pred)

        # 1. Low-confidence band accuracy
        if top_conf is not None:
            low = top_conf < thr
            if low.sum() > 0:
                low_acc = correct[low].mean()
                slices.append(
                    f"Low-confidence band (<{thr:.0%}): {low.sum()} samples, "
                    f"accuracy {low_acc:.0%}"
                )

        # 2. Per-class recall gaps (worst class)
        yt = pd.Series(np.asarray(y_test))
        per_class = {}
        for cls in yt.unique():
            mask = (yt == cls).values
            if mask.sum() >= 5:
                per_class[cls] = correct[mask].mean()
        if per_class:
            worst = min(per_class, key=per_class.get)
            if per_class[worst] < 0.70:
                slices.append(f"Class '{worst}': accuracy only {per_class[worst]:.0%}")

        # 3. Tabular subgroup slices (categorical columns)
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category"]).columns[:3]
            for col in cat_cols:
                for val in X[col].dropna().unique()[:20]:
                    mask = (X[col] == val).values
                    if mask.sum() >= 10:
                        a = correct[mask].mean()
                        if a < 0.60:
                            slices.append(f"{col}='{val}': accuracy {a:.0%} ({mask.sum()} samples)")
        return slices[:6]

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _trust_score(self, acc, conf_mean, uncertain, robustness, calib_gap) -> int:
        # Weighted blend, all on 0–1 then scaled to 100
        score = (
            0.40 * acc +
            0.25 * robustness +
            0.20 * (1 - calib_gap) +
            0.15 * (1 - uncertain)
        )
        return int(round(max(0.0, min(1.0, score)) * 100))

    def _regression_trust(self, model, X_test, y_test, notes) -> TrustReport:
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        robustness = self._robustness_reg(model, X_test, y_pred)
        trust = int(round(max(0.0, min(1.0, 0.6 * max(r2, 0) + 0.4 * robustness)) * 100))
        grade = ("production-ready" if trust >= 80 else
                 "needs review"     if trust >= 60 else "not ready")
        notes.append("Regression trust = blend of R² and prediction stability")
        report = TrustReport(
            trust_score=trust, grade=grade,
            confidence_mean=round(float(max(r2, 0)), 4), confidence_std=0.0,
            uncertain_fraction=0.0, robustness_score=round(robustness, 4),
            calibration_gap=round(1 - max(r2, 0), 4), failure_slices=[], notes=notes,
        )
        report.display()
        return report

    def _robustness_reg(self, model, X, base_pred) -> float:
        Xp = self._perturb(X, 0)
        if Xp is None:
            return 1.0
        try:
            p = model.predict(Xp)
            base = np.asarray(base_pred, dtype=float)
            denom = np.abs(base).mean() + 1e-9
            rel_change = np.abs(np.asarray(p, dtype=float) - base).mean() / denom
            return float(max(0.0, 1.0 - rel_change))
        except Exception:
            return 1.0


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_analyzer = TrustAnalyzer()


def analyze(model, X_test, y_test, task: Optional[str] = None) -> TrustReport:
    return _analyzer.analyze(model, X_test, y_test, task=task)
