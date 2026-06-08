"""Safe data correction engine — no silent modifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kaizenstat.utils.helpers import (
    detect_id_columns,
    get_categorical_cols,
    get_numeric_cols,
    risk_to_color,
    validate_dataframe,
)

console = Console()


@dataclass
class FixAction:
    column: str
    action: str
    reason: str
    risk_level: str      # LOW / MEDIUM / HIGH
    expected_impact: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixPlan:
    """Describes exactly what will be done before any data is modified."""

    actions: List[FixAction]
    safe: bool           # True = all actions are LOW risk

    def display(self) -> None:
        console.print(Panel.fit(
            f"[bold cyan]Fix Plan[/bold cyan]  │  "
            f"{len(self.actions)} action(s)  │  "
            f"Safe: {'[green]YES[/green]' if self.safe else '[yellow]MIXED[/yellow]'}",
            title="[bold]KaizenStat · Fix Preview[/bold]",
            border_style="cyan",
        ))

        if not self.actions:
            console.print("[bold green]✓ No fixes required.[/bold green]\n")
            return

        table = Table(box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Column", style="cyan", width=20)
        table.add_column("Action", width=22)
        table.add_column("Risk", justify="center", width=10)
        table.add_column("Reason")
        table.add_column("Expected Impact")

        for a in self.actions:
            rc = risk_to_color(a.risk_level)
            table.add_row(
                a.column,
                a.action,
                f"[{rc}]{a.risk_level}[/{rc}]",
                a.reason,
                a.expected_impact,
            )
        console.print(table)
        console.print("[dim]Call .apply(df) to execute this plan.[/dim]\n")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all planned actions and return a new (never modified-in-place) DataFrame."""
        result = df.copy()
        for action in self.actions:
            result = _apply_action(result, action)
        console.print(
            f"[bold green]✓ Applied {len(self.actions)} fix(es) → "
            f"output shape: {result.shape}[/bold green]"
        )
        return result


def _apply_action(df: pd.DataFrame, action: FixAction) -> pd.DataFrame:
    col = action.column
    act = action.action

    # drop_duplicates has no target column — must be checked before the column guard
    if act == "drop_duplicates":
        return df.drop_duplicates()

    if col not in df.columns:
        return df

    if act == "drop_column":
        return df.drop(columns=[col])

    if act == "fill_median":
        val = df[col].median()
        df = df.copy()
        df[col] = df[col].fillna(val)
        return df

    if act == "fill_mean":
        val = df[col].mean()
        df = df.copy()
        df[col] = df[col].fillna(val)
        return df

    if act == "fill_mode":
        mode = df[col].mode()
        val = mode.iloc[0] if not mode.empty else "Unknown"
        df = df.copy()
        df[col] = df[col].fillna(val)
        return df

    if act == "fill_constant":
        val = action.params.get("value", 0)
        df = df.copy()
        df[col] = df[col].fillna(val)
        return df

    if act == "clip_outliers":
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df = df.copy()
        df[col] = df[col].clip(lower=q1, upper=q3)
        return df

    if act == "log1p_transform":
        df = df.copy()
        df[col] = np.log1p(df[col].clip(lower=0))
        return df

    if act == "label_encode":
        df = df.copy()
        df[col] = df[col].astype("category").cat.codes
        return df

    if act == "drop_rows_missing_target":
        df = df.copy()
        df = df.dropna(subset=[col])
        return df

    return df


class FixEngine:
    """Plans and applies safe, explainable data corrections."""

    def plan(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        safe: bool = True,
    ) -> FixPlan:
        """Preview all suggested fixes without touching the data."""
        validate_dataframe(df, target)
        actions: List[FixAction] = []

        actions += self._plan_duplicates(df)
        actions += self._plan_missing_target(df, target)
        actions += self._plan_constant_cols(df, target)
        actions += self._plan_id_columns(df, target)
        actions += self._plan_missing_numeric(df, target)
        actions += self._plan_missing_categorical(df, target)
        actions += self._plan_categorical_encoding(df, target)
        actions += self._plan_outliers(df, target)

        if safe:
            actions = [a for a in actions if a.risk_level == "LOW"]

        plan = FixPlan(
            actions=actions,
            safe=all(a.risk_level == "LOW" for a in actions),
        )
        plan.display()
        return plan

    def apply(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        safe: bool = True,
    ) -> pd.DataFrame:
        """Plan and immediately apply fixes."""
        fix_plan = self.plan(df, target=target, safe=safe)
        return fix_plan.apply(df)

    def missing(self, df: pd.DataFrame, target: Optional[str] = None) -> FixPlan:
        validate_dataframe(df)
        actions = (
            self._plan_missing_target(df, target)
            + self._plan_missing_numeric(df, target)
            + self._plan_missing_categorical(df, target)
        )
        return FixPlan(actions=actions, safe=True)

    def outlier_handling(self, df: pd.DataFrame, target: Optional[str] = None) -> FixPlan:
        validate_dataframe(df)
        actions = self._plan_outliers(df, target)
        return FixPlan(actions=actions, safe=all(a.risk_level == "LOW" for a in actions))

    def encoding(self, df: pd.DataFrame, target: Optional[str] = None) -> FixPlan:
        validate_dataframe(df)
        cat_cols = get_categorical_cols(df, exclude=[target] if target else None)
        actions = [
            FixAction(
                column=col,
                action="label_encode",
                reason=f"Categorical feature with {df[col].nunique()} unique values",
                risk_level="LOW",
                expected_impact="Enables numeric models to use this feature",
            )
            for col in cat_cols
        ]
        return FixPlan(actions=actions, safe=True)

    def imbalance(self, df: pd.DataFrame, target: str) -> str:
        validate_dataframe(df, target)
        y = df[target].dropna()
        if y.empty:
            console.print("[yellow]⚠ Target column has no non-null values.[/yellow]")
            return "No data"
        counts = y.value_counts(normalize=True)
        if len(counts) < 2:
            console.print("[green]✓ Target has only one class — check labels.[/green]")
            return "Single class"
        minority = counts.iloc[-1]
        if minority < 0.1:
            msg = (
                f"Class imbalance detected (minority={minority:.1%}). "
                "Recommended: use class_weight='balanced' in your model, "
                "or apply SMOTE via imblearn.over_sampling.SMOTE."
            )
            console.print(f"[yellow]⚠ {msg}[/yellow]")
            return msg
        console.print("[green]✓ Classes are balanced.[/green]")
        return "Balanced"

    # ------------------------------------------------------------------ #
    # Plan builders                                                        #
    # ------------------------------------------------------------------ #

    def _plan_duplicates(self, df):
        n = df.duplicated().sum()
        if n == 0:
            return []
        return [FixAction(
            column="(all rows)",
            action="drop_duplicates",
            reason=f"{n:,} exact duplicate rows found",
            risk_level="LOW",
            expected_impact=f"Removes {n:,} rows; reduces dataset to {len(df) - n:,} rows",
        )]

    def _plan_missing_target(self, df, target):
        if not target or target not in df.columns:
            return []
        n = df[target].isnull().sum()
        if n == 0:
            return []
        return [FixAction(
            column=target,
            action="drop_rows_missing_target",
            reason=f"Target column has {n} missing values",
            risk_level="LOW",
            expected_impact=f"Removes {n} rows with unknown labels",
        )]

    def _plan_constant_cols(self, df, target):
        cols = [c for c in df.columns if df[c].nunique() <= 1 and c != target]
        return [
            FixAction(
                column=col,
                action="drop_column",
                reason="Zero-variance (constant) feature — no predictive value",
                risk_level="LOW",
                expected_impact="Removes noise, reduces dimensionality",
            )
            for col in cols
        ]

    def _plan_id_columns(self, df, target):
        id_cols = [c for c in detect_id_columns(df) if c != target]
        return [
            FixAction(
                column=col,
                action="drop_column",
                reason="ID-like column — leakage/noise risk",
                risk_level="LOW",
                expected_impact="Prevents model from memorising row identifiers",
            )
            for col in id_cols
        ]

    def _plan_missing_numeric(self, df, target):
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        actions = []
        for col in num_cols:
            pct = df[col].isnull().mean()
            if pct == 0:
                continue
            if pct > 0.50:
                actions.append(FixAction(
                    column=col,
                    action="drop_column",
                    reason=f"{pct:.0%} values missing (>50% threshold)",
                    risk_level="MEDIUM",
                    expected_impact="Removes unreliable feature",
                ))
            else:
                actions.append(FixAction(
                    column=col,
                    action="fill_median",
                    reason=f"{pct:.1%} missing values — median imputation",
                    risk_level="LOW",
                    expected_impact="Fills gaps without distorting distribution",
                ))
        return actions

    def _plan_missing_categorical(self, df, target):
        cat_cols = get_categorical_cols(df, exclude=[target] if target else None)
        actions = []
        for col in cat_cols:
            pct = df[col].isnull().mean()
            if pct == 0:
                continue
            if pct > 0.50:
                actions.append(FixAction(
                    column=col,
                    action="drop_column",
                    reason=f"{pct:.0%} values missing (>50% threshold)",
                    risk_level="MEDIUM",
                    expected_impact="Removes unreliable feature",
                ))
            else:
                actions.append(FixAction(
                    column=col,
                    action="fill_mode",
                    reason=f"{pct:.1%} missing values — mode imputation",
                    risk_level="LOW",
                    expected_impact="Fills gaps with most common category",
                ))
        return actions

    def _plan_categorical_encoding(self, df, target):
        """Label-encode object/string columns that still remain after missing-value fill."""
        already_id = {a.column for a in self._plan_id_columns(df, target)}
        cat_cols = get_categorical_cols(df, exclude=[target] if target else None)
        actions = []
        for col in cat_cols:
            if col in already_id:
                continue
            # Only encode columns that have at least 2 unique non-null values
            n_unique = df[col].dropna().nunique()
            if n_unique < 2:
                continue
            actions.append(FixAction(
                column=col,
                action="label_encode",
                reason=f"Categorical feature with {n_unique} unique values — encode to numeric",
                risk_level="LOW",
                expected_impact="Enables numeric models to use this feature directly",
            ))
        return actions

    def _plan_outliers(self, df, target):
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        actions = []
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            pct = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean()
            if pct > 0.01:   # aligned with health scorer threshold (>1% extreme values)
                actions.append(FixAction(
                    column=col,
                    action="clip_outliers",
                    reason=f"{pct:.1%} extreme outliers (>3×IQR) detected",
                    risk_level="MEDIUM",
                    expected_impact="Clips to [1%, 99%] percentile range",
                ))
        return actions


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_engine = FixEngine()


def plan(df: pd.DataFrame, target: Optional[str] = None, safe: bool = True) -> FixPlan:
    return _engine.plan(df, target=target, safe=safe)


def apply(df: pd.DataFrame, target: Optional[str] = None, safe: bool = True) -> pd.DataFrame:
    return _engine.apply(df, target=target, safe=safe)


def missing(df: pd.DataFrame, target: Optional[str] = None) -> FixPlan:
    return _engine.missing(df, target)


def outlier_handling(df: pd.DataFrame, target: Optional[str] = None) -> FixPlan:
    return _engine.outlier_handling(df, target)


def encoding(df: pd.DataFrame, target: Optional[str] = None) -> FixPlan:
    return _engine.encoding(df, target)


def imbalance(df: pd.DataFrame, target: str) -> str:
    return _engine.imbalance(df, target)
