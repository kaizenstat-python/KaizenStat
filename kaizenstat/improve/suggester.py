"""Rule-based improvement suggestion engine (AI-optional)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kaizenstat.utils.helpers import (
    detect_task_type,
    get_categorical_cols,
    get_numeric_cols,
    risk_to_color,
    validate_dataframe,
)

console = Console()


@dataclass
class Suggestion:
    priority: int            # 1 = highest
    category: str
    action: str
    reason: str
    expected_gain: str
    impact: str              # LOW / MEDIUM / HIGH


@dataclass
class ImprovementReport:
    suggestions: List[Suggestion]
    top_priority: Optional[Suggestion]

    def display(self) -> None:
        console.print(Panel.fit(
            f"[bold cyan]{len(self.suggestions)} improvement suggestion(s)[/bold cyan]",
            title="[bold]KaizenStat · Improvement Plan[/bold]",
            border_style="cyan",
        ))

        if not self.suggestions:
            console.print("[green]✓ No improvements found — system appears optimal.[/green]\n")
            return

        table = Table(box=box.ROUNDED, header_style="bold magenta")
        table.add_column("#", justify="center", width=4)
        table.add_column("Category", style="cyan", width=18)
        table.add_column("Impact", justify="center", width=10)
        table.add_column("Action")
        table.add_column("Expected Gain")

        for s in sorted(self.suggestions, key=lambda x: x.priority):
            ic = risk_to_color(s.impact)
            table.add_row(
                str(s.priority),
                s.category,
                f"[{ic}]{s.impact}[/{ic}]",
                s.action,
                s.expected_gain,
            )
        console.print(table)
        console.print()


class Suggester:
    """Generates prioritised, rule-based improvement suggestions."""

    def suggest(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        health_result: Optional[Any] = None,
        debug_result: Optional[Any] = None,
        validation_result: Optional[Any] = None,
    ) -> ImprovementReport:
        """Aggregate rule-based suggestions from all pipeline stages."""
        validate_dataframe(df, target)
        suggestions: List[Suggestion] = []
        priority = 1

        # 1. From health report
        if health_result is not None:
            suggestions += self._from_health(health_result, priority)
            priority += len(suggestions)

        # 2. From validation report
        if validation_result is not None:
            new = self._from_validation(validation_result, priority)
            suggestions += new
            priority += len(new)

        # 3. From debug result
        if debug_result is not None:
            new = self._from_debug(debug_result, priority)
            suggestions += new
            priority += len(new)

        # 4. Universal data-driven suggestions (debug_result used for tuning/importance hints)
        new = self._from_data(df, target, priority, debug_result=debug_result)
        suggestions += new

        # Re-number priorities in order
        for i, s in enumerate(suggestions, 1):
            s.priority = i

        top = suggestions[0] if suggestions else None
        report = ImprovementReport(suggestions=suggestions, top_priority=top)
        report.display()
        return report

    def prioritize(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Sort suggestions by impact (HIGH first) then by priority."""
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        return sorted(suggestions, key=lambda s: (order.get(s.impact, 9), s.priority))

    def feature_engineering(self, df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
        """Generate feature engineering ideas based on data characteristics."""
        validate_dataframe(df, target)
        ideas: List[str] = []
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)
        cat_cols = get_categorical_cols(df, exclude=[target] if target else None)

        # Skewed columns → log transform
        for col in num_cols:
            s = df[col].dropna()
            if len(s) > 0 and abs(s.skew()) > 2 and s.min() >= 0:
                ideas.append(f"Log-transform '{col}' (skew={s.skew():.2f}) → df['{col}_log'] = np.log1p(df['{col}'])")

        # Numeric pairs → ratio
        if len(num_cols) >= 2:
            for i in range(min(len(num_cols), 5)):
                for j in range(i + 1, min(len(num_cols), 5)):
                    a, b = num_cols[i], num_cols[j]
                    ideas.append(f"Ratio feature: df['{a}_div_{b}'] = df['{a}'] / (df['{b}'] + 1e-8)")

        # Categorical → count encoding
        for col in cat_cols:
            if df[col].nunique() > 5:
                ideas.append(f"Count-encode '{col}': df['{col}_count'] = df['{col}'].map(df['{col}'].value_counts())")

        if not ideas:
            ideas.append("No obvious feature engineering opportunities found in current dataset.")

        console.print(Panel.fit(
            "\n".join(f"  • {idea}" for idea in ideas[:10]),
            title="[bold]Feature Engineering Ideas[/bold]",
            border_style="cyan",
        ))
        return ideas

    # ------------------------------------------------------------------ #
    # Private rule evaluators                                              #
    # ------------------------------------------------------------------ #

    def _from_health(self, hr, start_priority):
        suggs = []
        for penalty in hr.penalties:
            action, gain = self._health_penalty_to_action(penalty.name)
            if action:
                suggs.append(Suggestion(
                    priority=start_priority + len(suggs),
                    category="Data Quality",
                    action=action,
                    reason=penalty.reason,
                    expected_gain=gain,
                    impact=penalty.risk_level,
                ))
        return suggs

    def _from_validation(self, vr, start_priority):
        suggs = []
        for iss in vr.issues:
            suggs.append(Suggestion(
                priority=start_priority + len(suggs),
                category="Validation",
                action=iss.recommendation,
                reason=iss.issue,
                expected_gain="Improved model reliability and generalization",
                impact=iss.risk_level,
            ))
        return suggs

    def _from_debug(self, dr, start_priority):
        suggs = []
        for iss in dr.issues:
            gap = getattr(dr, "gap", 0.0)
            test_score = getattr(dr, "test_score", 0.0)
            label = getattr(dr, "label", "")
            if "overfitting" in label and gap > 0:
                gain = f"Closing {gap:.0%} gap could recover +{min(int(gap * 80), 20)}% test accuracy"
            elif "underfitting" in label and test_score < 0.7:
                gain = f"Stronger model expected to add +{int((0.8 - test_score) * 100)}–{int((0.85 - test_score) * 100)}% accuracy"
            elif "leakage" in label:
                gain = "Removing leakage yields realistic score (likely -20–40% inflated accuracy)"
            else:
                gain = f"Expected +5–15% test score improvement (current: {test_score:.0%})"
            suggs.append(Suggestion(
                priority=start_priority + len(suggs),
                category="Model Debug",
                action=iss.suggestion,
                reason=iss.description,
                expected_gain=gain,
                impact=iss.risk_level,
            ))
        return suggs

    def _from_data(self, df, target, start_priority, debug_result=None):
        suggs = []
        num_cols = get_numeric_cols(df, exclude=[target] if target else None)

        ts = getattr(debug_result, "test_score", None)
        label = getattr(debug_result, "label", "")

        # Not enough data
        if len(df) < 1000:
            suggs.append(Suggestion(
                priority=start_priority + len(suggs),
                category="Data Volume",
                action="Collect more training data (target ≥ 2,000 rows minimum)",
                reason=f"Dataset has only {len(df):,} rows — small datasets overfit and have unreliable CV scores",
                expected_gain="Typically +5–15% test score improvement per 2× data increase",
                impact="HIGH",
            ))

        if target and target in df.columns:
            y    = df[target].dropna()
            task = detect_task_type(y)

            if task == "classification":
                counts = y.value_counts(normalize=True)
                minority = counts.iloc[-1] if len(counts) >= 2 else 1.0

                # Class imbalance → SMOTE
                if minority < 0.20:
                    recall_gain = min(30, int((0.20 - minority) * 100))
                    suggs.append(Suggestion(
                        priority=start_priority + len(suggs),
                        category="Class Imbalance",
                        action="Apply SMOTE (imbalanced-learn) or set class_weight='balanced'",
                        reason=f"Minority class is only {minority:.0%} — model predicts majority class by default",
                        expected_gain=f"Expected +{recall_gain}–{recall_gain + 10}% minority-class recall and F1",
                        impact="HIGH",
                    ))

                # Low test score → stacking ensemble
                if ts is not None and ts < 0.80 and "leakage" not in label:
                    gap_to_80 = max(0, int((0.80 - ts) * 100))
                    suggs.append(Suggestion(
                        priority=start_priority + len(suggs),
                        category="Ensemble / AutoML",
                        action="Run doctor.train_auto(tune=True, ensemble=True) to build a stacking ensemble",
                        reason=f"Test score {ts:.2f} is below 0.80 — stacking diverse models typically closes the gap",
                        expected_gain=f"Typically +{max(3, gap_to_80 // 3)}–{max(8, gap_to_80 // 2)}% accuracy gain from stacking + tuning",
                        impact="HIGH",
                    ))

                # Low test score → hyperparameter tuning
                elif ts is not None and ts < 0.85 and "leakage" not in label:
                    suggs.append(Suggestion(
                        priority=start_priority + len(suggs),
                        category="Model Tuning",
                        action="Run doctor.train(tune=True) for 2-stage progressive hyperparameter search",
                        reason=f"Test score {ts:.2f} has room to improve via better hyperparameters",
                        expected_gain="Typically +3–10% accuracy gain (progressive search beats single random search)",
                        impact="HIGH",
                    ))

        # Model calibration — suggest when probability estimates are needed
        if ts is not None and ts > 0.70 and "leakage" not in label:
            suggs.append(Suggestion(
                priority=start_priority + len(suggs),
                category="Calibration",
                action="Call doctor.trust_score() to check calibration gap — apply Platt scaling if gap > 0.05",
                reason="Good accuracy ≠ reliable probabilities; overconfident models fail in production",
                expected_gain="More reliable confidence scores (+0.03–0.08 calibration gap reduction typical)",
                impact="MEDIUM",
            ))

        # Feature importance → drop near-zero features
        if (debug_result is not None
                and debug_result.feature_importances is not None
                and not debug_result.feature_importances.empty):
            low = int((debug_result.feature_importances < 0.01).sum())
            if low > 0:
                suggs.append(Suggestion(
                    priority=start_priority + len(suggs),
                    category="Feature Selection",
                    action=f"Drop {low} near-zero importance feature(s) to reduce noise",
                    reason="Near-zero importance features add noise and slow training without improving accuracy",
                    expected_gain=f"Simpler model, faster inference; often +1–3% generalization gain",
                    impact="MEDIUM",
                ))

        # Missing values → KNN imputation
        miss_rate = df.isnull().mean().mean()
        if miss_rate > 0.05:
            suggs.append(Suggestion(
                priority=start_priority + len(suggs),
                category="Data Completeness",
                action="Apply KNNImputer or IterativeImputer instead of simple median/mode fill",
                reason=f"Average missing rate is {miss_rate:.1%} — richer imputation preserves feature relationships",
                expected_gain="+1–4% accuracy vs median fill on datasets with > 10% missing values",
                impact="MEDIUM",
            ))

        # Skewed features → log transform
        if num_cols:
            skewed = int((df[num_cols].skew().abs() > 2).sum())
            if skewed > 0:
                suggs.append(Suggestion(
                    priority=start_priority + len(suggs),
                    category="Feature Engineering",
                    action=f"Apply log1p or Yeo-Johnson transform to {skewed} highly skewed feature(s)",
                    reason="Skewness > 2 hurts linear models and distance-based algorithms",
                    expected_gain="+2–6% for logistic regression and SVM; tree models are less affected",
                    impact="MEDIUM",
                ))

        # Failure clustering actions
        if debug_result is not None:
            for iss in getattr(debug_result, "issues", []):
                if "Failure Slice" in iss.name:
                    col = iss.name.replace("Failure Slice: ", "")
                    suggs.append(Suggestion(
                        priority=start_priority + len(suggs),
                        category="Subgroup Fix",
                        action=f"Collect more labelled samples for the failing subgroup in '{col}'",
                        reason=iss.description,
                        expected_gain="Closing subgroup accuracy gap typically adds +3–8% overall F1",
                        impact="HIGH",
                    ))

        return suggs

    @staticmethod
    def _health_penalty_to_action(penalty_name: str):
        mapping = {
            "Missing Values":    ("Impute or drop high-missing columns using fix.missing()",
                                  "Reduces information loss"),
            "Duplicate Rows":    ("Remove duplicates using fix.apply()",
                                  "Prevents training data bias"),
            "Class Imbalance":   ("Apply SMOTE or class_weight='balanced'",
                                  "Improves minority class recall"),
            "Outliers":          ("Clip extreme outliers using fix.outlier_handling()",
                                  "Reduces model sensitivity to extreme values"),
            "High Skewness":     ("Apply log1p/Box-Cox transformation",
                                  "Normalises distributions for linear models"),
            "Constant Features": ("Drop zero-variance features using fix.plan()",
                                  "Removes noise from feature space"),
            "High Cardinality":  ("Apply target encoding or frequency encoding",
                                  "Reduces dimensionality from OHE explosion"),
            "Leakage Risk":      ("Investigate and remove leakage features immediately",
                                  "Prevents inflated train scores / production failure"),
        }
        return mapping.get(penalty_name, (None, None))


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_suggester = Suggester()


def suggest(
    df: pd.DataFrame,
    target: Optional[str] = None,
    health_result=None,
    debug_result=None,
    validation_result=None,
) -> ImprovementReport:
    return _suggester.suggest(
        df, target,
        health_result=health_result,
        debug_result=debug_result,
        validation_result=validation_result,
    )


def prioritize(suggestions: List[Suggestion]) -> List[Suggestion]:
    return _suggester.prioritize(suggestions)


def feature_engineering(df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    return _suggester.feature_engineering(df, target)
