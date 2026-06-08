"""Statistical assumption and data leakage validation engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kaizenstat.utils.helpers import (
    get_numeric_cols,
    risk_to_color,
    validate_dataframe,
)

console = Console()


@dataclass
class ValidationIssue:
    check: str
    issue: str
    reason: str
    risk_level: str        # LOW / MEDIUM / HIGH
    recommendation: str
    column: Optional[str] = None


@dataclass
class ValidationReport:
    passed: bool
    issues: List[ValidationIssue]
    checks_run: int

    def display(self) -> None:
        status = (
            "[bold green]✓ PASSED[/bold green]" if self.passed
            else "[bold red]✗ ISSUES FOUND[/bold red]"
        )
        console.print(Panel.fit(
            f"Validation: {status}\n"
            f"Checks: {self.checks_run}  │  Issues: {len(self.issues)}",
            title="[bold]KaizenStat · Validation Report[/bold]",
            border_style="cyan",
        ))

        if not self.issues:
            console.print("[bold green]✓ All checks passed.[/bold green]\n")
            return

        table = Table(box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Check", style="cyan", width=22)
        table.add_column("Column", width=18)
        table.add_column("Risk", justify="center", width=10)
        table.add_column("Issue")
        table.add_column("Recommendation")

        for iss in self.issues:
            rc = risk_to_color(iss.risk_level)
            table.add_row(
                iss.check,
                iss.column or "[dim]—[/dim]",
                f"[{rc}]{iss.risk_level}[/{rc}]",
                iss.issue,
                iss.recommendation,
            )
        console.print(table)
        console.print()


class Validator:
    """Validates statistical assumptions before modeling."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def assumptions(self, df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
        """Run all assumption checks in sequence."""
        validate_dataframe(df, target)
        issues: List[ValidationIssue] = []
        issues += self._check_skewness(df, target)
        issues += self._check_multicollinearity(df, target)
        if target:
            issues += self._check_leakage(df, target)
        issues += self._check_distribution(df, target)
        issues += self._check_normality(df, target)
        return ValidationReport(
            passed=not any(i.risk_level == "HIGH" for i in issues),
            issues=issues,
            checks_run=5,
        )

    def skewness(self, df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
        validate_dataframe(df)
        iss = self._check_skewness(df, target)
        return ValidationReport(passed=not iss, issues=iss, checks_run=1)

    def multicollinearity(self, df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
        validate_dataframe(df)
        iss = self._check_multicollinearity(df, target)
        return ValidationReport(passed=not iss, issues=iss, checks_run=1)

    def leakage(self, df: pd.DataFrame, target: str) -> ValidationReport:
        validate_dataframe(df, target)
        iss = self._check_leakage(df, target)
        return ValidationReport(passed=not iss, issues=iss, checks_run=1)

    def distribution_check(self, df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
        validate_dataframe(df)
        iss = self._check_distribution(df, target)
        return ValidationReport(passed=not iss, issues=iss, checks_run=1)

    def detect_drift(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, float]:
        """
        Detect distribution drift between train and test sets using the KS test.

        Returns a dict of {column: p_value} for columns that show statistically
        significant drift (p < 0.05). An empty dict means no drift detected.
        """
        from scipy.stats import ks_2samp

        drifted: Dict[str, float] = {}
        num_cols = [c for c in X_train.select_dtypes(include="number").columns
                    if c in X_test.columns]

        for col in num_cols:
            try:
                _, p = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
                if p < 0.05:
                    drifted[col] = round(p, 6)
            except Exception:
                pass

        if drifted:
            drift_lines = "\n".join(
                f"  • {col:<25s} p={p:.4f}  ← distribution shifted"
                for col, p in sorted(drifted.items(), key=lambda x: x[1])
            )
            console.print(Panel.fit(
                f"[bold red]{len(drifted)} feature(s) show distribution shift between train and test[/bold red]\n"
                + drift_lines,
                title="[bold]KaizenStat · Drift Detection[/bold]",
                border_style="red",
            ))
        else:
            console.print(Panel.fit(
                "[bold green]✓ No significant distribution drift detected[/bold green]",
                title="[bold]KaizenStat · Drift Detection[/bold]",
                border_style="green",
            ))

        return drifted

    # ------------------------------------------------------------------ #
    # Private checkers                                                     #
    # ------------------------------------------------------------------ #

    def _check_normality(self, df, target):
        issues = []
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        non_normal = []
        for col in num_cols[:20]:
            s = df[col].dropna()
            if len(s) < 8:
                continue
            sample = s.sample(min(len(s), 4000), random_state=42)
            try:
                if len(sample) <= 5000:
                    _, p = stats.shapiro(sample)
                else:
                    _, p = stats.normaltest(sample)  # pragma: no cover
                if p < 0.01:
                    non_normal.append(col)
            except Exception:
                pass

        if non_normal:
            issues.append(ValidationIssue(
                check="Normality",
                issue=f"{len(non_normal)} features significantly non-normal (p < 0.01)",
                reason="Shapiro-Wilk / D'Agostino test rejected normality",
                risk_level="MEDIUM",
                recommendation="Apply log/Box-Cox transform or use tree-based models",
                column=", ".join(non_normal[:3]) + ("…" if len(non_normal) > 3 else ""),
            ))
        return issues

    def _check_skewness(self, df, target):
        issues = []
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        if not num_cols:
            return issues

        skew = df[num_cols].skew().abs()
        severe = skew[skew > 3]
        moderate = skew[(skew > 1) & (skew <= 3)]

        if not severe.empty:
            issues.append(ValidationIssue(
                check="Skewness",
                issue=f"{len(severe)} features with |skew| > 3  (severe)",
                reason="Heavy-tailed distributions distort linear/distance models",
                risk_level="HIGH",
                recommendation="Apply log1p, sqrt, or Box-Cox transformation",
                column=", ".join(severe.index[:3].tolist()),
            ))
        if not moderate.empty:
            issues.append(ValidationIssue(
                check="Skewness",
                issue=f"{len(moderate)} features with 1 < |skew| ≤ 3  (moderate)",
                reason="Mild skew may affect model calibration",
                risk_level="LOW",
                recommendation="Consider log or sqrt transformation",
                column=", ".join(moderate.index[:3].tolist()),
            ))
        return issues

    def _check_multicollinearity(self, df, target):
        issues = []
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        if len(num_cols) < 2:
            return issues

        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high_pairs = [
            (col, row, corr.loc[row, col])
            for col in upper.columns
            for row in upper.index
            if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.90
        ]

        if high_pairs:
            pairs_str = "; ".join(f"{a}↔{b}({c:.2f})" for a, b, c in high_pairs[:3])
            issues.append(ValidationIssue(
                check="Multicollinearity",
                issue=f"{len(high_pairs)} feature pairs with correlation > 0.90",
                reason="High collinearity inflates variance and hurts interpretability",
                risk_level="HIGH" if len(high_pairs) > 3 else "MEDIUM",
                recommendation="Drop one from each correlated pair or use PCA/regularization",
                column=pairs_str,
            ))

        # Simplified VIF
        try:
            from sklearn.linear_model import LinearRegression
            clean = df[num_cols].dropna()
            sample = clean.sample(min(len(clean), 2000), random_state=42)
            high_vif = []
            if len(sample) > len(num_cols):
                for col in num_cols:
                    others = [c for c in num_cols if c != col]
                    if not others:
                        continue  # pragma: no cover
                    X_v = sample[others].fillna(0)
                    y_v = sample[col].fillna(0)
                    r2 = LinearRegression().fit(X_v, y_v).score(X_v, y_v)
                    r2 = min(r2, 0.9999)
                    if r2 > 0:
                        vif = 1 / (1 - r2)
                        if vif > 10:
                            high_vif.append((col, vif))
            if high_vif:
                issues.append(ValidationIssue(
                    check="VIF",
                    issue=f"{len(high_vif)} features with VIF > 10",
                    reason="Near-perfect linear dependency detected",
                    risk_level="HIGH",
                    recommendation="Remove collinear features or apply PCA",
                    column=", ".join(f"{c}(VIF={v:.0f})" for c, v in high_vif[:3]),
                ))
        except Exception:
            pass

        return issues

    def _check_leakage(self, df, target):
        issues = []
        y_num = pd.to_numeric(df[target], errors="coerce")
        num_cols = get_numeric_cols(df, exclude=[target])
        suspects = []
        leak_names = []

        for col in num_cols:
            try:
                c = abs(df[col].corr(y_num))
                if c > 0.98:
                    suspects.append(f"{col}(r={c:.3f})")
                    leak_names.append(col)
            except Exception:
                pass

        # Unique-value proxy: flag non-target object/string columns where every row is unique (ID-like).
        # Continuous numeric columns (float) with all-unique values are normal and NOT leakage.
        n_rows = len(df)
        for col in df.columns:
            if col == target:
                continue
            if col in leak_names:  # already flagged by correlation check
                continue
            col_dtype = df[col].dtype
            is_string_col = col_dtype == object or str(col_dtype) == "string"
            is_integer_col = pd.api.types.is_integer_dtype(col_dtype)
            if (is_string_col or is_integer_col) and n_rows > 10 and df[col].nunique() == n_rows:
                suspects.append(f"{col}(unique-key)")
                leak_names.append(col)

        if suspects:
            console.print(
                f"[bold red]🚨 Leakage detected in: {leak_names}[/bold red]\n"
                f"   These features are nearly identical to the target — your accuracy is fake.\n"
                f"   Remove them before training or your model will fail in production."
            )
            issues.append(ValidationIssue(
                check="Data Leakage",
                issue=f"{len(suspects)} potential leakage feature(s)",
                reason="Feature nearly identical to target → inflated train scores",
                risk_level="HIGH",
                recommendation="Investigate and remove leaking features before training",
                column=", ".join(suspects[:3]),
            ))
        return issues

    def _check_distribution(self, df, target):
        issues = []
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)

        near_const = []
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 4 or s.std() == 0:
                continue
            cv = abs(s.std() / (s.mean() + 1e-9))
            if cv < 0.005:
                near_const.append(col)

        if near_const:
            issues.append(ValidationIssue(
                check="Near-Constant",
                issue=f"{len(near_const)} features with CV < 0.5%",
                reason="Extremely low variance — may add noise rather than signal",
                risk_level="LOW",
                recommendation="Consider dropping unless domain-significant",
                column=", ".join(near_const[:3]),
            ))
        return issues


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_validator = Validator()


def assumptions(df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
    return _validator.assumptions(df, target)


def skewness(df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
    return _validator.skewness(df, target)


def multicollinearity(df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
    return _validator.multicollinearity(df, target)


def leakage(df: pd.DataFrame, target: str) -> ValidationReport:
    return _validator.leakage(df, target)


def distribution_check(df: pd.DataFrame, target: Optional[str] = None) -> ValidationReport:
    return _validator.distribution_check(df, target)


def detect_drift(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, float]:
    return _validator.detect_drift(X_train, X_test)
