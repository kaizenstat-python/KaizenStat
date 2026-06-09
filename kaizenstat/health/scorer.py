"""Data Health Scoring Engine — the core measurement layer of KaizenStat."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kaizenstat.utils.helpers import (
    get_categorical_cols,
    get_numeric_cols,
    risk_to_color,
    score_to_grade,
    validate_dataframe,
)

console = Console()


@dataclass
class HealthPenalty:
    name: str
    penalty: float       # always negative (e.g. -12.0)
    reason: str
    risk_level: str      # LOW / MEDIUM / HIGH / CRITICAL


@dataclass
class HealthResult:
    score: float
    grade: str
    risk_level: str
    penalties: List[HealthPenalty]
    summary: str
    rows: int
    columns: int

    def __repr__(self) -> str:
        return (
            f"HealthResult(score={self.score}/100, grade={self.grade}, "
            f"risk={self.risk_level}, penalties={len(self.penalties)})"
        )

    def display(self) -> None:
        color = risk_to_color(self.risk_level)
        console.print(Panel.fit(
            f"[bold cyan]Data Health Score: {self.score} / 100   Grade: {self.grade}[/bold cyan]\n"
            f"Risk Level: [{color}]{self.risk_level}[/{color}]  "
            f"│  {self.rows:,} rows × {self.columns} columns",
            title="[bold]KaizenStat · Data Health Report[/bold]",
            border_style="cyan",
        ))

        if self.penalties:
            table = Table(box=box.ROUNDED, header_style="bold magenta")
            table.add_column("Issue", style="cyan", width=22)
            table.add_column("Penalty", justify="right", style="bold red", width=9)
            table.add_column("Risk", justify="center", width=10)
            table.add_column("Reason")

            for p in self.penalties:
                rc = risk_to_color(p.risk_level)
                table.add_row(
                    p.name,
                    f"{p.penalty:+.1f}",
                    f"[{rc}]{p.risk_level}[/{rc}]",
                    p.reason,
                )
            console.print(table)
        else:
            console.print("[bold green]✓ No penalties — dataset is in excellent health![/bold green]")

        console.print(f"\n[dim]Summary:[/dim] {self.summary}\n")


class HealthScorer:
    """Compute a 0–100 health score for any pandas DataFrame."""

    # Maximum penalty per category
    _LIMITS = {
        "missing":       20.0,
        "duplicates":    10.0,
        "imbalance":     20.0,
        "outliers":      10.0,
        "skewness":      10.0,
        "constant":       5.0,
        "high_card":      8.0,
        "leakage":       20.0,
    }

    def score(self, df: pd.DataFrame, target: Optional[str] = None) -> float:
        """Return a single 0–100 health score."""
        return self.breakdown(df, target).score

    def report(self, df: pd.DataFrame, target: Optional[str] = None) -> HealthResult:
        """Compute and print a rich health report."""
        result = self.breakdown(df, target)
        result.display()
        return result

    def breakdown(self, df: pd.DataFrame, target: Optional[str] = None) -> HealthResult:
        """Compute the full penalty breakdown and return a HealthResult."""
        validate_dataframe(df, target)
        penalties: List[HealthPenalty] = []
        total = 0.0

        total += self._missing(df, target, penalties)
        total += self._duplicates(df, penalties)
        total += self._imbalance(df, target, penalties)
        total += self._outliers(df, target, penalties)
        total += self._skewness(df, target, penalties)
        total += self._constant_features(df, target, penalties)
        total += self._high_cardinality(df, target, penalties)
        total += self._leakage_proxy(df, target, penalties)

        score = round(max(0.0, 100.0 - total), 1)
        grade = score_to_grade(score)
        risk = (
            "CRITICAL" if score < 40 else
            "HIGH"     if score < 60 else
            "MEDIUM"   if score < 80 else
            "LOW"
        )

        return HealthResult(
            score=score,
            grade=grade,
            risk_level=risk,
            penalties=penalties,
            summary=self._summary(score),
            rows=len(df),
            columns=len(df.columns),
        )

    # ------------------------------------------------------------------ #
    # Internal penalty calculators                                         #
    # ------------------------------------------------------------------ #

    def _missing(self, df, target, out):
        by_col = df.isnull().mean()
        n_miss = (by_col > 0).sum()
        if n_miss == 0:
            return 0.0
        avg, worst = by_col.mean(), by_col.max()
        deduct = min(self._LIMITS["missing"], worst * 80 + avg * 40)
        risk = "HIGH" if worst > 0.30 else "MEDIUM" if worst > 0.05 else "LOW"
        worst_col = by_col.idxmax()
        out.append(HealthPenalty(
            name="Missing Values",
            penalty=-round(deduct, 1),
            reason=f"{n_miss} cols affected; worst: '{worst_col}' ({worst:.0%})",
            risk_level=risk,
        ))
        return deduct

    def _duplicates(self, df, out):
        n = df.duplicated().sum()
        if n == 0:
            return 0.0
        pct = n / len(df)
        deduct = min(self._LIMITS["duplicates"], pct * 60)
        risk = "HIGH" if pct > 0.15 else "MEDIUM"
        out.append(HealthPenalty(
            name="Duplicate Rows",
            penalty=-round(deduct, 1),
            reason=f"{n:,} duplicate rows ({pct:.1%} of data)",
            risk_level=risk,
        ))
        return deduct

    def _imbalance(self, df, target, out):
        if not target or target not in df.columns:
            return 0.0
        y = df[target].dropna()
        # Skip imbalance check for regression targets (continuous numeric with many unique values)
        # and for text/string targets with too many classes to be a classification problem.
        n_unique = y.nunique()
        if pd.api.types.is_float_dtype(y) and n_unique > 20:
            return 0.0
        if pd.api.types.is_integer_dtype(y) and n_unique > 50:
            return 0.0
        if y.dtype == "object" and n_unique > 100:
            return 0.0
        counts = y.value_counts(normalize=True)
        if len(counts) <= 1:
            return 0.0
        minority = counts.iloc[-1]
        if minority >= 0.1:
            return 0.0
        deduct = min(self._LIMITS["imbalance"], (0.1 - minority) * 200)
        risk = "HIGH" if minority < 0.05 else "MEDIUM"
        out.append(HealthPenalty(
            name="Class Imbalance",
            penalty=-round(deduct, 1),
            reason=f"Minority class '{counts.index[-1]}' = {minority:.1%} of samples",
            risk_level=risk,
        ))
        return deduct

    def _outliers(self, df, target, out):
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        n_outlier_cols = 0
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            if ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean() > 0.01:
                n_outlier_cols += 1
        if n_outlier_cols == 0:
            return 0.0
        deduct = min(self._LIMITS["outliers"], n_outlier_cols * 1.5)
        out.append(HealthPenalty(
            name="Outliers",
            penalty=-round(deduct, 1),
            reason=f"{n_outlier_cols} numeric features with >1% extreme values (>3×IQR)",
            risk_level="MEDIUM",
        ))
        return deduct

    def _skewness(self, df, target, out):
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        if not num_cols:
            return 0.0
        skew = df[num_cols].skew().abs().dropna()  # dropna guards against all-NaN columns
        if skew.empty:
            return 0.0
        high = (skew > 3).sum()
        if high == 0:
            return 0.0
        deduct = min(self._LIMITS["skewness"], high * 1.5)
        out.append(HealthPenalty(
            name="High Skewness",
            penalty=-round(deduct, 1),
            reason=f"{high} features with |skewness| > 3 (distorts linear models)",
            risk_level="MEDIUM",
        ))
        return deduct

    def _constant_features(self, df, target, out):
        cols = [c for c in df.columns if df[c].nunique() <= 1 and c != target]
        if not cols:
            return 0.0
        deduct = min(self._LIMITS["constant"], len(cols) * 1.5)
        names = ", ".join(cols[:3]) + ("..." if len(cols) > 3 else "")
        out.append(HealthPenalty(
            name="Constant Features",
            penalty=-round(deduct, 1),
            reason=f"{len(cols)} zero-variance features: {names}",
            risk_level="LOW",
        ))
        return deduct

    def _high_cardinality(self, df, target, out):
        cat_cols = get_categorical_cols(df, exclude=[target] if target else None)
        high = [c for c in cat_cols if df[c].nunique() > 50]
        if not high:
            return 0.0
        deduct = min(self._LIMITS["high_card"], len(high) * 2.5)
        out.append(HealthPenalty(
            name="High Cardinality",
            penalty=-round(deduct, 1),
            reason=f"{len(high)} categoricals with >50 unique values (encoding risk)",
            risk_level="MEDIUM",
        ))
        return deduct

    def _leakage_proxy(self, df, target, out):
        if not target or target not in df.columns:
            return 0.0
        y_num = pd.to_numeric(df[target], errors="coerce")
        num_cols = get_numeric_cols(df, exclude=[target])
        suspects = []
        for col in num_cols:
            try:
                corr = abs(df[col].corr(y_num))
                if corr > 0.98:
                    suspects.append(col)
            except Exception:
                pass
        if not suspects:
            return 0.0
        deduct = self._LIMITS["leakage"]
        out.append(HealthPenalty(
            name="Leakage Proxy",
            penalty=-round(deduct, 1),
            reason=f"Correlation >0.98 with target: {', '.join(suspects[:3])}",
            risk_level="HIGH",
        ))
        return deduct

    @staticmethod
    def _summary(score: float) -> str:
        if score >= 90:
            return "Excellent health. Dataset is ready for modeling."
        if score >= 75:
            return "Good health. Address minor issues before training."
        if score >= 60:
            return "Moderate issues detected. Fix recommended before modeling."
        if score >= 40:
            return "Significant problems. Training will likely yield poor results."
        return "Critical state. Major intervention required before use."


# ------------------------------------------------------------------ #
# Module-level convenience API: from kaizenstat import health; health.score(df)
# ------------------------------------------------------------------ #
_scorer = HealthScorer()


def score(df: pd.DataFrame, target: Optional[str] = None) -> float:
    return _scorer.score(df, target)


def report(df: pd.DataFrame, target: Optional[str] = None) -> HealthResult:
    return _scorer.report(df, target)


def breakdown(df: pd.DataFrame, target: Optional[str] = None) -> HealthResult:
    return _scorer.breakdown(df, target)
