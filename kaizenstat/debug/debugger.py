"""ML Model Debugger — priority-based diagnosis engine."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from kaizenstat.utils.helpers import detect_task_type, risk_to_color

warnings.filterwarnings("ignore")
console = Console()


# ------------------------------------------------------------------ #
# Fix suggestion library (keyed by label)
# ------------------------------------------------------------------ #
_FIX_SUGGESTIONS: Dict[str, List[str]] = {
    "data_leakage": [
        "Check train-test split",
        "Remove data leakage features",
        "Use proper cross-validation",
    ],
    "leakage_risk": [
        "Check train-test split for contamination",
        "Audit features for post-event or target-correlated columns",
        "Validate with a completely held-out dataset",
    ],
    "data_issue": [
        "Verify train-test split — test score cannot exceed train score in typical settings",
        "Check for dataset shuffling issues",
        "Inspect for distribution mismatch between train and test",
    ],
    "excellent": [
        "Verify on a fresh holdout set to confirm generalisation",
        "Test on edge cases and distribution shifts",
    ],
    "healthy": [
        "Run cross-validation to confirm stability",
        "Monitor performance on production data",
    ],
    "acceptable": [
        "Explore feature engineering to improve test score",
        "Run cross-validation to confirm stability",
    ],
    "overfitting_risk": [
        "Add mild regularization",
        "Monitor with cross-validation",
        "Consider pruning or early stopping",
    ],
    "overfitting": [
        "Add regularization",
        "Reduce model complexity",
        "Increase training data",
    ],
    "severe_overfitting": [
        "Add strong regularization (L1/L2)",
        "Drastically reduce model complexity (depth, n_estimators)",
        "Collect significantly more training data",
        "Use cross-validation to estimate true generalisation",
    ],
    "underfitting": [
        "Increase model complexity",
        "Add better features",
        "Train longer",
    ],
    "severe_underfitting": [
        "Increase model complexity significantly",
        "Add better features",
        "Train longer",
        "Verify target labels are correct",
    ],
    "weak_model": [
        "Improve feature engineering",
        "Try different model architecture",
    ],
    "broken_model": [
        "Overhaul feature engineering pipeline",
        "Try a fundamentally different model architecture",
        "Check for data quality issues and label noise",
        "Verify the task framing is correct",
    ],
    "unstable_model": [
        "Use cross-validation",
        "Check data consistency",
    ],
}

# Human-readable label → diagnosis (backward-compat for DataDoctor)
_LABEL_TO_DIAGNOSIS: Dict[str, str] = {
    "data_leakage":       "data_issue",
    "leakage_risk":       "data_issue",
    "data_issue":         "data_issue",
    "excellent":          "healthy",
    "healthy":            "healthy",
    "acceptable":         "healthy",
    "overfitting_risk":   "overfitting",
    "overfitting":        "overfitting",
    "severe_overfitting": "overfitting",
    "underfitting":       "underfitting",
    "severe_underfitting":"underfitting",
    "weak_model":         "data_issue",
    "broken_model":       "data_issue",
    "unstable_model":     "data_issue",
}

_LABEL_COLOR: Dict[str, str] = {
    "data_leakage":       "red",
    "leakage_risk":       "red",
    "data_issue":         "red",
    "excellent":          "green",
    "healthy":            "green",
    "acceptable":         "cyan",
    "overfitting_risk":   "yellow",
    "overfitting":        "red",
    "severe_overfitting": "red",
    "underfitting":       "yellow",
    "severe_underfitting":"red",
    "weak_model":         "red",
    "broken_model":       "red",
    "unstable_model":     "yellow",
}

_SEVERITY_COLOR: Dict[str, str] = {
    "LOW":      "green",
    "MEDIUM":   "yellow",
    "HIGH":     "orange3",
    "CRITICAL": "red",
}


@dataclass
class DebugIssue:
    name: str
    description: str
    root_cause: str
    risk_level: str
    suggestion: str


@dataclass
class DebugResult:
    task: str
    train_score: float
    test_score: float
    gap: float
    diagnosis: str          # backward-compat: overfitting/underfitting/healthy/data_issue
    root_cause: str
    issues: List[DebugIssue]
    suggestions: List[str]
    feature_importances: Optional[pd.Series] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    # Extended fields from the priority classifier
    avg_score: float = 0.0
    label: str = ""
    severity: str = ""
    confidence: float = 0.0
    health_score: int = 0
    why_bullets: List[str] = field(default_factory=list)

    def display(self) -> None:
        label_color    = _LABEL_COLOR.get(self.label, "white")
        severity_color = _SEVERITY_COLOR.get(self.severity, "white")
        gap_color      = "red" if abs(self.gap) > 0.10 else "yellow" if abs(self.gap) > 0.05 else "green"

        console.print(Panel.fit(
            f"[bold]Train:[/bold] {self.train_score:.4f}   "
            f"[bold]Test:[/bold] {self.test_score:.4f}   "
            f"[bold]Gap:[/bold] [{gap_color}]{self.gap:+.4f}[/{gap_color}]   "
            f"[bold]Avg:[/bold] {self.avg_score:.4f}\n"
            f"[bold]Label:[/bold] [{label_color}]{self.label}[/{label_color}]   "
            f"[bold]Severity:[/bold] [{severity_color}]{self.severity}[/{severity_color}]   "
            f"[bold]Confidence:[/bold] {self.confidence:.0%}   "
            f"[bold]Health Score:[/bold] {self.health_score}/100\n"
            f"[bold]Root Cause:[/bold] {self.root_cause}",
            title="[bold]KaizenStat · Model Debug Report[/bold]",
            border_style="cyan",
        ))

        if self.issues:
            table = Table(box=box.ROUNDED, header_style="bold magenta")
            table.add_column("Issue", style="cyan", width=24)
            table.add_column("Risk", justify="center", width=10)
            table.add_column("Description")
            table.add_column("Suggestion")
            for iss in self.issues:
                rc = risk_to_color(iss.risk_level)
                table.add_row(
                    iss.name,
                    f"[{rc}]{iss.risk_level}[/{rc}]",
                    iss.description,
                    iss.suggestion,
                )
            console.print(table)

        if self.why_bullets:
            console.print("\n[bold yellow]Why is the model behaving this way?[/bold yellow]")
            for b in self.why_bullets:
                console.print(f"  {b}")

        if self.suggestions:
            console.print("\n[bold cyan]Actionable Suggestions:[/bold cyan]")
            for i, s in enumerate(self.suggestions, 1):
                console.print(f"  {i}. {s}")

        if self.feature_importances is not None and not self.feature_importances.empty:
            console.print("\n[bold cyan]Top Feature Importances:[/bold cyan]")
            top = self.feature_importances.head(10)
            table2 = Table(box=box.SIMPLE, show_header=False)
            table2.add_column("Feature", style="cyan")
            table2.add_column("Score", justify="right")
            for feat, val in top.items():
                table2.add_row(str(feat), f"{val:.4f}")
            console.print(table2)
        console.print()


class ModelDebugger:
    """Priority-based ML model diagnosis engine."""

    def model_failure(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> DebugResult:
        """Full failure analysis using priority-based classification."""
        task        = detect_task_type(y_test)
        train_score = model.score(X_train, y_train)
        test_score  = model.score(X_test, y_test)
        gap         = train_score - test_score
        avg_score   = (train_score + test_score) / 2

        label, severity, confidence = self._classify(train_score, test_score, gap)
        health_score = self._compute_health_score(label, gap, test_score)
        reasoning    = self._build_reasoning(label, train_score, test_score, gap, avg_score)
        why_bullets  = self._why_bullets(label, train_score, test_score, gap, y_train, y_test, X_train, X_test)
        fix_suggestions = _FIX_SUGGESTIONS.get(label, ["Use cross-validation", "Check data consistency"])

        # Single structured issue from the classifier
        issues: List[DebugIssue] = [
            DebugIssue(
                name=label.replace("_", " ").title(),
                description=f"Train={train_score:.3f}  Test={test_score:.3f}  Gap={gap:+.3f}  Avg={avg_score:.3f}",
                root_cause=reasoning,
                risk_level=severity,
                suggestion=fix_suggestions[0],
            )
        ]

        # Class imbalance check — appended as supplemental issue
        if task == "classification":
            issues += self._diagnose_imbalance(y_train, y_test, model, X_test)

        # Data vs model blame — tells users WHERE to focus effort
        blame      = self._data_vs_model_blame(label, test_score,
                                                X_train, y_train, task)
        if blame:
            issues.append(blame)

        # Failure clustering — which subgroups fail most
        if task == "classification" and isinstance(X_test, pd.DataFrame):
            issues += self._failure_clustering(model, X_test, y_test)

        diagnosis = _LABEL_TO_DIAGNOSIS.get(label, "data_issue")
        fi        = self._extract_feature_importance(model, X_test, y_test, task)
        metrics   = self._compute_extra_metrics(model, X_test, y_test, task)

        result = DebugResult(
            task=task,
            train_score=round(train_score, 4),
            test_score=round(test_score, 4),
            gap=round(gap, 4),
            diagnosis=diagnosis,
            root_cause=reasoning,
            issues=issues,
            suggestions=fix_suggestions,
            feature_importances=fi,
            metrics=metrics,
            avg_score=round(avg_score, 4),
            label=label,
            severity=severity,
            confidence=confidence,
            health_score=health_score,
            why_bullets=why_bullets,
        )
        result.display()
        return result

    def overfitting_check(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Quick overfitting check with train/test gap analysis."""
        train_score = model.score(X_train, y_train)
        test_score  = model.score(X_test, y_test)
        gap         = train_score - test_score

        label, severity, confidence = self._classify(train_score, test_score, gap)
        is_overfitting = "overfitting" in label

        result = {
            "train_score": round(train_score, 4),
            "test_score":  round(test_score, 4),
            "gap":         round(gap, 4),
            "label":       label,
            "is_overfitting": is_overfitting,
            "severity":    severity,
            "confidence":  confidence,
        }

        sev_color = _SEVERITY_COLOR.get(severity, "white")
        gap_color = "red" if abs(gap) > 0.10 else "yellow" if abs(gap) > 0.05 else "green"
        console.print(Panel.fit(
            f"Train: [green]{train_score:.4f}[/green]  │  "
            f"Test: [bold]{test_score:.4f}[/bold]  │  "
            f"Gap: [{gap_color}]{gap:+.4f}[/{gap_color}]\n"
            f"Label: [{_LABEL_COLOR.get(label, 'white')}]{label}[/{_LABEL_COLOR.get(label, 'white')}]  │  "
            f"Severity: [{sev_color}]{severity}[/{sev_color}]  │  "
            f"Confidence: {confidence:.0%}",
            title="[bold]Overfitting Check[/bold]",
            border_style="cyan",
        ))
        return result

    def error_analysis(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Return a DataFrame of misclassified / high-error samples."""
        task   = detect_task_type(y_test)
        y_pred = model.predict(X_test)

        result = X_test.copy()
        result["y_true"] = y_test.values
        result["y_pred"] = y_pred

        if task == "classification":
            result["correct"] = result["y_true"] == result["y_pred"]
            errors = result[~result["correct"]]
            console.print(f"[cyan]Error Analysis:[/cyan] {len(errors)} misclassified "
                          f"out of {len(result)} ({len(errors)/len(result):.1%})")
        else:
            result["residual"]  = result["y_true"] - result["y_pred"]
            result["abs_error"] = result["residual"].abs()
            errors = result.nlargest(min(20, len(result)), "abs_error")
            console.print(f"[cyan]Error Analysis:[/cyan] Top {len(errors)} highest-error samples")

        return errors

    def feature_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> pd.Series:
        """Extract and display feature importances."""
        names = feature_names or list(X_test.columns)
        task  = detect_task_type(y_test)
        fi    = self._extract_feature_importance(model, X_test, y_test, task, names)

        if fi is not None and not fi.empty:
            console.print("[bold cyan]Feature Importances (top 15):[/bold cyan]")
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
            table.add_column("Rank", justify="center", width=6)
            table.add_column("Feature", style="cyan")
            table.add_column("Importance", justify="right")
            for i, (feat, val) in enumerate(fi.head(15).items(), 1):
                table.add_row(str(i), str(feat), f"{val:.4f}")
            console.print(table)
            return fi

        console.print("[yellow]Feature importances not available for this model.[/yellow]")
        return pd.Series(dtype=float)

    def feature_impact(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 15,
    ) -> Dict[str, float]:
        """
        Counterfactual feature impact: measure how much accuracy drops when each
        feature is removed one at a time.

        Returns a dict {feature: accuracy_drop} sorted descending.
        High values = feature is critical. Negative = feature was hurting the model.
        """
        task     = detect_task_type(y)
        baseline = model.score(X, y)

        impacts: Dict[str, float] = {}
        for col in X.columns:
            X_temp = X.drop(columns=[col])
            try:
                score = model.score(X_temp, y)
                impacts[col] = round(baseline - score, 4)
            except Exception:
                impacts[col] = 0.0

        sorted_impacts = dict(
            sorted(impacts.items(), key=lambda kv: kv[1], reverse=True)
        )

        console.print(Panel.fit(
            f"[bold cyan]Baseline score: {baseline:.4f}[/bold cyan]  │  "
            f"Task: {task}  │  Features tested: {len(impacts)}",
            title="[bold]KaizenStat · Counterfactual Feature Impact[/bold]",
            border_style="cyan",
        ))
        from rich import box as _box
        from rich.table import Table as _Table
        table = _Table(box=_box.SIMPLE, show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="center", width=6)
        table.add_column("Feature", style="cyan")
        table.add_column("Score Drop", justify="right")
        table.add_column("Verdict", width=22)
        for i, (feat, drop) in enumerate(list(sorted_impacts.items())[:top_n], 1):
            color   = "red" if drop > 0.05 else "yellow" if drop > 0.01 else "green"
            verdict = "CRITICAL" if drop > 0.05 else "Useful" if drop > 0.01 else "Low impact"
            table.add_row(str(i), feat, f"[{color}]{drop:+.4f}[/{color}]", verdict)
        console.print(table)
        console.print()

        return sorted_impacts

    def dataset_difficulty(
        self,
        X: pd.DataFrame,
        y,
        cv: int = 3,
    ) -> float:
        """
        Estimate how hard the dataset is for a baseline model (0 = trivially easy, 1 = impossible).

        Uses LogisticRegression/Ridge cross-val score.
        Difficulty = 1 − baseline_score.
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # Accept both numpy arrays and pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        task = detect_task_type(y)
        X_num = X.select_dtypes(include="number").fillna(0)
        if X_num.shape[1] == 0:
            console.print("[yellow]dataset_difficulty: no numeric columns found — returning 0.5[/yellow]")
            return 0.5

        try:
            if task == "classification":
                estimator = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=500))])
                scoring   = "accuracy"
            else:
                estimator = Pipeline([("s", StandardScaler()), ("m", Ridge())])
                scoring   = "r2"

            scores   = cross_val_score(estimator, X_num, y, cv=min(cv, 3), scoring=scoring)
            baseline = float(np.clip(scores.mean(), 0, 1))
        except Exception:
            baseline = 0.5

        difficulty = round(1.0 - baseline, 4)
        color      = "red" if difficulty > 0.5 else "yellow" if difficulty > 0.25 else "green"
        label      = "Hard" if difficulty > 0.5 else "Moderate" if difficulty > 0.25 else "Easy"
        console.print(Panel.fit(
            f"[bold {color}]Difficulty: {difficulty:.2f} — {label}[/bold {color}]\n"
            f"  Baseline ({task}): {baseline:.4f}  │  "
            f"Difficulty = 1 − baseline_score",
            title="[bold]KaizenStat · Dataset Difficulty[/bold]",
            border_style=color,
        ))
        return difficulty

    def recommend_actions(
        self,
        profile: Dict[str, Any],
        debug_result: Any,
    ) -> List[str]:
        """
        Generate a prioritised what-to-do list based on the data profile and debug result.

        Args:
            profile:      Output of ModelTrainer._analyze_data()
            debug_result: DebugResult from model_failure()

        Returns a list of actionable strings, ranked by expected impact.
        """
        if profile is None:
            profile = {}

        actions: List[str] = []
        label = getattr(debug_result, "label", "")

        # --- Critical: leakage ---
        if "leakage" in label:
            actions.append("🚨 CRITICAL: Remove leaking features — your accuracy is inflated and will collapse in production")

        # --- Imbalance ---
        if profile.get("imbalance", 1.0) < 0.20:
            pct = f"{profile['imbalance']:.0%}"
            actions.append(
                f"⚡ Class imbalance ({pct} minority) → apply SMOTE or class_weight='balanced' "
                f"(expected +10–20% F1)"
            )

        # --- Overfitting ---
        if "overfitting" in label:
            gap = getattr(debug_result, "gap", 0)
            actions.append(
                f"⚠️  Overfitting (gap={gap:+.3f}) → add regularization, reduce model complexity, "
                f"or collect more training data"
            )

        # --- Underfitting ---
        if "underfitting" in label:
            actions.append(
                "📉 Underfitting → use boosting models (XGBoost/LightGBM), add feature interactions, "
                "or run train(tune=True)"
            )

        # --- Low test score → tuning ---
        ts = getattr(debug_result, "test_score", 1.0)
        if ts < 0.75 and "leakage" not in label:
            actions.append(
                f"🔧 Test score {ts:.2f} has room to improve → run train_auto(tune=True) "
                f"for RandomizedSearchCV (typically +5–15%)"
            )

        # --- High dimensionality ---
        if profile.get("high_dim", False):
            actions.append(
                f"📐 High-dimensional data ({profile['n_cols']} features) → "
                f"train_auto() already applies SelectKBest; consider PCA or feature importance pruning"
            )

        # --- Missing values ---
        if profile.get("missing_ratio", 0) > 0.10:
            rate = f"{profile['missing_ratio']:.0%}"
            actions.append(
                f"❓ {rate} missing values → apply KNN or iterative imputation before training"
            )

        # --- Small dataset ---
        if profile.get("n_rows", 10000) < 1000:
            actions.append(
                f"📊 Only {profile['n_rows']} rows → collect more data; "
                f"cross-validation estimates are unreliable on small datasets"
            )

        # --- Near-zero importance features ---
        fi = getattr(debug_result, "feature_importances", None)
        if fi is not None and not getattr(fi, "empty", True):
            low = int((fi < 0.01).sum())
            if low > 0:
                actions.append(
                    f"🗑️  {low} near-zero importance feature(s) → drop them to reduce noise and speed up training"
                )

        if not actions:
            actions.append("✅ No critical actions required — model looks healthy")

        console.print(Panel.fit(
            "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions)),
            title="[bold]KaizenStat · Recommended Actions[/bold]",
            border_style="cyan",
        ))
        return actions

    def bias_detection(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Check for performance disparities across subgroups."""
        task   = detect_task_type(y_test)
        y_pred = model.predict(X_test)

        if not sensitive_features:
            cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
            sensitive_features = cat_cols[:3]

        if not sensitive_features:
            console.print("[yellow]No categorical features found for bias detection.[/yellow]")
            return {}

        report: Dict[str, Dict[str, float]] = {}
        console.print("[bold cyan]Bias Detection:[/bold cyan]")

        for feat in sensitive_features:
            if feat not in X_test.columns:
                continue
            groups: Dict[str, float] = {}
            for val in X_test[feat].unique():
                mask = X_test[feat] == val
                if mask.sum() < 5:
                    continue
                score = (accuracy_score(y_test[mask], y_pred[mask])
                         if task == "classification" else r2_score(y_test[mask], y_pred[mask]))
                groups[str(val)] = round(score, 4)

            if groups:
                report[feat] = groups
                vals   = list(groups.values())
                spread = max(vals) - min(vals)
                color  = "red" if spread > 0.10 else "yellow" if spread > 0.05 else "green"
                console.print(
                    f"  [cyan]{feat}[/cyan] — spread: [{color}]{spread:.4f}[/{color}]  "
                    + "  ".join(f"{k}={v}" for k, v in groups.items())
                )

        return report

    # ------------------------------------------------------------------ #
    # Core classifier — implements the 8-level priority spec              #
    # ------------------------------------------------------------------ #

    def _classify(
        self, train: float, test: float, gap: float
    ) -> Tuple[str, str, float]:
        """
        Priority-based classification (top-to-bottom, first match wins).

        Level order:
          1. Critical data issues   (leakage, test > train)
          2. Strong generalisation  (small gap, high test)
          3. Early overfitting      (gap 0.05–0.10)
          4. Overfitting            (gap 0.10–0.20)
          5. Severe overfitting     (gap > 0.20)
          6. Underfitting           (both scores low)
          8. Fallback
          7. Override               (applied last, can override any level)

        Returns (label, severity, confidence).
        """
        # --- Level 1: Critical data issues ---
        if train == 1.0 and test == 1.0:
            label, severity, confidence = "data_leakage", "CRITICAL", 0.99
        elif train >= 0.98 and test >= 0.98:
            label, severity, confidence = "leakage_risk", "HIGH", 0.95
        elif test > train:
            label, severity, confidence = "data_issue", "CRITICAL", 0.98

        # --- Level 6: Underfitting (checked before gap levels to avoid masking) ---
        elif train <= 0.60 and test <= 0.60:
            label, severity, confidence = "severe_underfitting", "CRITICAL", 0.95
        elif train <= 0.70 and test <= 0.70:
            label, severity, confidence = "underfitting", "HIGH", 0.90

        # --- Level 2: Strong generalisation ---
        elif gap <= 0.05 and test >= 0.90:
            label, severity, confidence = "excellent", "LOW", 0.95
        elif gap <= 0.05 and test >= 0.80:
            label, severity, confidence = "healthy", "LOW", 0.90
        elif gap <= 0.05:
            label, severity, confidence = "acceptable", "LOW", 0.80

        # --- Level 3: Early overfitting ---
        elif gap <= 0.10:   # 0.05 < gap <= 0.10
            label, severity, confidence = "overfitting_risk", "MEDIUM", 0.75

        # --- Level 4: Overfitting ---
        elif gap <= 0.20:   # 0.10 < gap <= 0.20
            label, severity, confidence = "overfitting", "HIGH", 0.85

        # --- Level 5: Severe overfitting ---
        elif gap > 0.20:
            label, severity, confidence = "severe_overfitting", "CRITICAL", 0.95

        # --- Level 8: Fallback ---
        else:  # pragma: no cover
            label, severity, confidence = "unstable_model", "MEDIUM", 0.60

        # --- Level 7: Override logic (applied after primary classification) ---
        if gap > 0.30 and test < 0.60:
            return "broken_model", "CRITICAL", 0.98
        if gap > 0.15 and test < 0.70:
            return "weak_model", "HIGH", 0.90

        return label, severity, confidence

    # ------------------------------------------------------------------ #
    # Health score                                                         #
    # ------------------------------------------------------------------ #

    def _compute_health_score(self, label: str, gap: float, test: float) -> int:
        score = 100

        label_penalties = {
            "data_leakage":       -80,
            "leakage_risk":       -50,
            "severe_overfitting": -40,
            "overfitting":        -25,
            "overfitting_risk":   -10,
            "underfitting":       -30,
            "severe_underfitting":-50,
            "weak_model":         -35,
            "broken_model":       -70,
            "acceptable":         -10,
        }
        score += label_penalties.get(label, 0)

        # Gap penalty
        if gap > 0.20:
            score -= 20
        elif gap > 0.10:
            score -= 10

        # Low test score penalty
        if test < 0.60:
            score -= 25
        elif test < 0.70:
            score -= 15

        return max(0, min(100, score))

    # ------------------------------------------------------------------ #
    # Root-cause AI — human-readable bullet explanation                   #
    # ------------------------------------------------------------------ #

    def _why_bullets(self, label, train, test, gap, _y_train, y_test, X_train, X_test) -> List[str]:
        bullets: List[str] = []
        task = detect_task_type(y_test)

        if "overfitting" in label:
            bullets.append(f"• Train-test gap is {gap:.2f} — model memorised training data instead of learning patterns")
            ratio = len(X_train) / max(X_test.shape[1], 1)
            if ratio < 10:
                bullets.append(f"• Only {ratio:.0f} training rows per feature — too few samples for the model's complexity")
        elif "underfitting" in label:
            bullets.append(f"• Both train ({train:.2f}) and test ({test:.2f}) are low — model is too simple for this data")
            bullets.append("• A linear model on non-linear data, or insufficient features, are common causes")
        elif label in ("data_leakage", "leakage_risk"):
            bullets.append("• Scores are suspiciously perfect — a feature likely encodes the target directly")
            bullets.append("• Check for columns computed from the target, or timestamps that leak future information")
        elif label == "data_issue":
            bullets.append(f"• Test ({test:.2f}) > Train ({train:.2f}) — this should not happen with a proper random split")
            bullets.append("• Likely cause: shuffling error, train/test contamination, or distribution mismatch")

        if task == "classification":
            counts = y_test.value_counts(normalize=True)
            if len(counts) >= 2 and counts.iloc[-1] < 0.20:
                bullets.append(
                    f"• Class imbalance: minority class is {counts.iloc[-1]:.0%} of test data — "
                    "model biased toward majority class"
                )

        miss_rate = X_test.isnull().mean().mean()
        if miss_rate > 0.10:
            bullets.append(f"• {miss_rate:.0%} of test features are missing — imputation quality affects predictions")

        low_var = int((X_train.select_dtypes(include="number").var() < 0.01).sum())
        if low_var > 0:
            bullets.append(f"• {low_var} near-constant feature(s) detected — these add noise without signal")

        return bullets

    # ------------------------------------------------------------------ #
    # Reasoning builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_reasoning(
        self, label: str, train: float, test: float, gap: float, avg: float
    ) -> str:
        reasons = {
            "data_leakage":
                f"Both train ({train:.3f}) and test ({test:.3f}) are exactly 1.0 — "
                "this is statistically impossible without data leakage or a trivially simple dataset. "
                "The model has zero generalisation challenge; target information is likely encoded in a feature.",
            "leakage_risk":
                f"Train ({train:.3f}) and test ({test:.3f}) are both ≥ 0.98 with a gap of {gap:.3f}. "
                "Near-identical perfect scores suggest leakage or severe overfitting to a non-representative test set. "
                "Validate generalisation on a completely separate holdout.",
            "data_issue":
                f"Test score ({test:.3f}) exceeds train score ({train:.3f}) — gap {gap:+.3f}. "
                "In a properly split dataset the train score should be at least as high as test. "
                "Likely causes: data shuffling error, train/test contamination, or distribution mismatch.",
            "excellent":
                f"Gap {gap:.3f} is minimal and test score {test:.3f} indicates strong generalisation capacity. "
                "The model learned meaningful signal without memorisation.",
            "healthy":
                f"Gap {gap:.3f} is within acceptable range and test score {test:.3f} is solid. "
                "Model generalises well — no significant overfitting or underfitting detected.",
            "acceptable":
                f"Gap {gap:.3f} is small but test score {test:.3f} leaves room for improvement. "
                "Model is not overfitting but capacity or features may be limiting performance.",
            "overfitting_risk":
                f"Gap {gap:.3f} (train={train:.3f}, test={test:.3f}) is in the early overfitting range. "
                "The model is starting to memorise training data — regularisation is recommended before it worsens.",
            "overfitting":
                f"Gap {gap:.3f} (train={train:.3f}, test={test:.3f}) confirms overfitting. "
                "The model memorised training patterns that do not generalise; reduce complexity or add regularisation.",
            "severe_overfitting":
                f"Gap {gap:.3f} (train={train:.3f}, test={test:.3f}) is severe. "
                "The model has drastically memorised training data and cannot generalise. "
                "Fundamental architectural changes or significantly more data are required.",
            "underfitting":
                f"Both train ({train:.3f}) and test ({test:.3f}) are ≤ 0.70 with a gap of {gap:.3f}. "
                "The model lacks capacity to capture patterns in the data — it is underfitting.",
            "severe_underfitting":
                f"Both train ({train:.3f}) and test ({test:.3f}) are ≤ 0.60. "
                "The model is severely underfitting and performing near or below chance level. "
                "Consider a fundamentally more powerful architecture or better features.",
            "weak_model":
                f"Gap {gap:.3f} combined with low test score {test:.3f} indicates the model is both overfitting "
                "and performing poorly. Feature engineering and model selection need to be revisited.",
            "broken_model":
                f"Gap {gap:.3f} and test score {test:.3f} indicate a broken model — extreme overfitting "
                "with near-random test performance. The model has completely failed to learn generalisable patterns.",
            "unstable_model":
                f"Train={train:.3f}, test={test:.3f}, gap={gap:.3f}, avg={avg:.3f}. "
                "No clear failure pattern detected but performance is inconsistent — "
                "use cross-validation to identify stability issues.",
        }
        return reasons.get(label, f"Train={train:.3f}, test={test:.3f}, gap={gap:.3f}, avg={avg:.3f}.")

    # ------------------------------------------------------------------ #
    # Supplemental checks                                                  #
    # ------------------------------------------------------------------ #

    def _data_vs_model_blame(
        self,
        label: str,
        test_score: float,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task: str,
    ) -> Optional[DebugIssue]:
        """
        Distinguish DATA problems from MODEL problems.

        Heuristic: fit a strong baseline (XGBoost/RandomForest) on the same
        training data.  If the baseline also performs poorly on test, the
        problem is in the data (noise, missing signal, bad labels).
        If the baseline does well, the problem is in the current model.
        """
        if label in ("excellent", "healthy"):
            return None

        if not isinstance(X_train, pd.DataFrame):
            return None

        try:
            from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline as _P

            X_num = X_train.select_dtypes(include="number").fillna(0)
            if X_num.shape[1] == 0:
                return None

            if task == "classification":
                baseline = _P([("s", StandardScaler()),
                                ("m", RFC(n_estimators=50, random_state=42, n_jobs=-1))])
            else:
                baseline = _P([("s", StandardScaler()),
                                ("m", RFR(n_estimators=50, random_state=42, n_jobs=-1))])

            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            cv_obj = (StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                      if task == "classification" else
                      KFold(n_splits=3, shuffle=True, random_state=42))
            metric = "accuracy" if task == "classification" else "r2"
            scores = cross_val_score(baseline, X_num, y_train, cv=cv_obj, scoring=metric)
            baseline_cv = float(scores.mean())

            if baseline_cv < 0.60 and test_score < 0.65:
                return DebugIssue(
                    name="Data Problem (not Model)",
                    description=(
                        f"Even a RandomForest baseline scores only {baseline_cv:.2f} CV — "
                        f"the data itself lacks predictive signal"
                    ),
                    root_cause=(
                        "Features do not explain the target well. "
                        "Likely causes: noisy labels, weak features, wrong task framing."
                    ),
                    risk_level="CRITICAL",
                    suggestion=(
                        "Collect better features or verify label quality — "
                        "swapping models will not help"
                    ),
                )

            if baseline_cv >= 0.70 and test_score < baseline_cv - 0.10:
                return DebugIssue(
                    name="Model Problem (data is OK)",
                    description=(
                        f"RandomForest baseline achieves {baseline_cv:.2f} CV but "
                        f"current model only scores {test_score:.2f} on test"
                    ),
                    root_cause=(
                        "The data has signal but the current model is not capturing it. "
                        "Model complexity, regularisation, or feature encoding needs adjustment."
                    ),
                    risk_level="HIGH",
                    suggestion=(
                        "Try train_auto(tune=True, ensemble=True) to switch to a stronger model"
                    ),
                )

        except Exception:
            pass

        return None

    def _failure_clustering(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[DebugIssue]:
        """
        Identify WHICH subgroups the model fails on most.

        For each categorical column, computes per-group accuracy and flags
        groups where accuracy is more than 15 percentage points below average.
        Output example: "city='NY': accuracy 52% vs overall 81%"
        """
        if X_test.empty:
            return []

        try:
            y_pred    = model.predict(X_test)
            overall   = float(accuracy_score(y_test, y_pred))
            cat_cols  = X_test.select_dtypes(include=["object", "category"]).columns.tolist()

            # Also bucket any low-cardinality numeric column
            num_low   = [c for c in X_test.select_dtypes(include="number").columns
                         if X_test[c].nunique() <= 8]
            check_cols = (cat_cols + num_low)[:5]   # cap at 5 to avoid noise

            issues: List[DebugIssue] = []
            for col in check_cols:
                failing_groups = []
                for val in X_test[col].unique():
                    mask = X_test[col] == val
                    if mask.sum() < 5:
                        continue
                    grp_acc = float(accuracy_score(y_test[mask], y_pred[mask]))
                    if overall - grp_acc > 0.15:
                        failing_groups.append((str(val), grp_acc, int(mask.sum())))

                if failing_groups:
                    worst = min(failing_groups, key=lambda x: x[1])
                    desc  = "  ".join(
                        f"{col}='{v}': {a:.0%} ({n} samples)"
                        for v, a, n in sorted(failing_groups, key=lambda x: x[1])[:3]
                    )
                    issues.append(DebugIssue(
                        name=f"Failure Slice: {col}",
                        description=f"Overall {overall:.0%} but {desc}",
                        root_cause=(
                            f"Model underperforms on specific values of '{col}'. "
                            f"Worst: '{col}={worst[0]}' → {worst[1]:.0%}"
                        ),
                        risk_level="HIGH",
                        suggestion=(
                            f"Investigate '{col}' subgroup — consider target encoding or "
                            f"collecting more samples for the failing groups"
                        ),
                    ))

            return issues

        except Exception:
            return []

    def _diagnose_imbalance(self, _y_train, y_test, model, X_test) -> List[DebugIssue]:
        issues = []
        counts = y_test.value_counts(normalize=True)
        if len(counts) > 1 and counts.iloc[-1] < 0.15:
            y_pred = model.predict(X_test)
            try:
                f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                acc = accuracy_score(y_test, y_pred)
                if acc - f1 > 0.10:
                    issues.append(DebugIssue(
                        name="Class Imbalance",
                        description=f"Accuracy {acc:.3f} >> F1 {f1:.3f} — minority class ignored",
                        root_cause="Model predicting majority class for most samples",
                        risk_level="HIGH",
                        suggestion="Use class_weight='balanced', SMOTE, or threshold tuning",
                    ))
            except Exception:
                pass
        return issues

    # ------------------------------------------------------------------ #
    # Feature importance                                                   #
    # ------------------------------------------------------------------ #

    def _extract_feature_importance(self, model, X_test, y_test, task, names=None):
        inner = model
        if hasattr(model, "named_steps"):
            inner = model.named_steps.get("model", model)

        feat_names = self._get_feature_names(model, X_test)

        # Tree-based models: use built-in importances
        if hasattr(inner, "feature_importances_"):
            try:
                fi = inner.feature_importances_
                if len(feat_names) == len(fi):
                    return pd.Series(fi, index=feat_names).sort_values(ascending=False)
            except Exception:
                pass

        # Linear models: use |coef_| — handle multiclass (coef_ is 2-D)
        if hasattr(inner, "coef_"):
            try:
                coef = np.abs(np.asarray(inner.coef_))
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)   # average across classes
                coef = coef.ravel()
                if len(feat_names) == len(coef):
                    return pd.Series(coef, index=feat_names).sort_values(ascending=False)
            except Exception:
                pass

        # Fallback: permutation importance (model-agnostic, always correct shape)
        try:
            scoring = "accuracy" if task == "classification" else "r2"
            pi = permutation_importance(
                model, X_test, y_test,
                scoring=scoring, n_repeats=5, random_state=42, n_jobs=-1,
            )
            pi_names = feat_names if len(feat_names) == len(pi.importances_mean) \
                       else list(X_test.columns)
            return pd.Series(pi.importances_mean, index=pi_names).sort_values(ascending=False)
        except Exception:
            return None

    def _get_feature_names(self, model, X_test):
        """Return feature names that match the transformer output length."""
        if hasattr(model, "named_steps") and "prep" in model.named_steps:
            prep = model.named_steps["prep"]
            # Try get_feature_names_out() (sklearn 1.0+)
            try:
                return list(prep.get_feature_names_out())
            except Exception:
                pass
            # Fallback: infer count from a dummy transform
            try:
                transformed = prep.transform(X_test.iloc[:1])
                n = transformed.shape[1]
                return [f"feature_{i}" for i in range(n)]
            except Exception:
                pass
        return list(X_test.columns)

    def _compute_extra_metrics(self, model, X_test, y_test, task):
        y_pred  = model.predict(X_test)
        metrics = {}
        try:
            if task == "classification":
                metrics["accuracy"]    = round(accuracy_score(y_test, y_pred), 4)
                metrics["f1_weighted"] = round(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
                )
            else:
                metrics["r2"]   = round(r2_score(y_test, y_pred), 4)
                metrics["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)
                metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        except Exception:
            pass
        return metrics


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_debugger = ModelDebugger()


def model_failure(model, X_train, X_test, y_train, y_test) -> DebugResult:
    return _debugger.model_failure(model, X_train, X_test, y_train, y_test)


def overfitting_check(model, X_train, X_test, y_train, y_test) -> Dict:
    return _debugger.overfitting_check(model, X_train, X_test, y_train, y_test)


def error_analysis(model, X_test, y_test) -> pd.DataFrame:
    return _debugger.error_analysis(model, X_test, y_test)


def feature_importance(model, X_test, y_test, feature_names=None) -> pd.Series:
    return _debugger.feature_importance(model, X_test, y_test, feature_names)


def bias_detection(model, X_test, y_test, sensitive_features=None) -> Dict:
    return _debugger.bias_detection(model, X_test, y_test, sensitive_features)


def feature_impact(model, X, y, top_n: int = 15) -> Dict[str, float]:
    return _debugger.feature_impact(model, X, y, top_n=top_n)


def dataset_difficulty(X, y, cv: int = 3) -> float:
    return _debugger.dataset_difficulty(X, y, cv=cv)


def recommend_actions(profile: Dict, debug_result) -> List[str]:
    return _debugger.recommend_actions(profile, debug_result)
