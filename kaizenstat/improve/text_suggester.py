"""Text improvement engine — NLP counterpart of suggester.py.

Reuses ImprovementReport / Suggestion. Rules are data- and debug-aware:
  • sparse matrix         → n-grams / char n-grams
  • weak performance      → embeddings
  • class imbalance       → balancing / oversampling
  • small dataset         → augmentation
  • low test score        → tune=True
"""
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from kaizenstat.improve.suggester import ImprovementReport, Suggestion
from kaizenstat.utils.helpers import (
    detect_task_type, dominant_text_column, validate_dataframe,
)


class TextSuggester:
    """Generates prioritised, rule-based NLP improvement suggestions."""

    def suggest(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        text_col: Optional[str] = None,
        health_result: Optional[Any] = None,
        debug_result: Optional[Any] = None,
        validation_result: Optional[Any] = None,
    ) -> ImprovementReport:
        validate_dataframe(df, target)
        text_col = text_col or dominant_text_column(df, exclude=[target] if target else None)
        suggestions: List[Suggestion] = []

        if validation_result is not None:
            suggestions += self._from_validation(validation_result)
        if debug_result is not None:
            suggestions += self._from_debug(debug_result)
        suggestions += self._from_data(df, target, text_col)

        # Dedupe by action text, keep first (highest-priority) occurrence
        seen, deduped = set(), []
        for s in suggestions:
            if s.action in seen:
                continue
            seen.add(s.action)
            deduped.append(s)

        for i, s in enumerate(deduped, 1):
            s.priority = i

        top = deduped[0] if deduped else None
        report = ImprovementReport(suggestions=deduped, top_priority=top)
        report.display()
        return report

    # ------------------------------------------------------------------ #
    # Rule sources                                                        #
    # ------------------------------------------------------------------ #

    def _from_validation(self, vr) -> List[Suggestion]:
        out = []
        for iss in getattr(vr, "issues", []):
            out.append(Suggestion(
                priority=1, category="Text Validation",
                action=iss.recommendation,
                reason=iss.issue,
                expected_gain="Cleaner features, better generalisation",
                impact=iss.risk_level,
            ))
        return out

    def _from_debug(self, dr) -> List[Suggestion]:
        out = []
        label = getattr(dr, "label", "")
        test_score = getattr(dr, "test_score", 1.0)

        # Mine the structured text issues
        for iss in getattr(dr, "issues", []):
            name = iss.name.lower()
            if "sparse" in name:
                out.append(Suggestion(
                    priority=1, category="Representation",
                    action="Add char n-grams (analyzer='char_wb', ngram_range=(2,5)) and lower min_df",
                    reason="TF-IDF matrix is extremely sparse",
                    expected_gain="Denser features, +3–10% on short/noisy text",
                    impact="HIGH",
                ))
            elif "weak representation" in name:
                out.append(Suggestion(
                    priority=2, category="Representation",
                    action="Use ngram_range=(1,2) or (1,3) to capture phrases",
                    reason="Unigram-only / tiny vocabulary",
                    expected_gain="Captures context, +2–8% accuracy",
                    impact="MEDIUM",
                ))
            elif "rare-token" in name:
                out.append(Suggestion(
                    priority=1, category="Regularisation",
                    action="Set min_df=3–5 to prune one-off tokens + increase regularisation",
                    reason="Overfitting to rare tokens",
                    expected_gain="Smaller gap, better test score",
                    impact="HIGH",
                ))
            elif "imbalance" in name:
                out.append(Suggestion(
                    priority=1, category="Class Imbalance",
                    action="Keep class_weight='balanced'; oversample minority or adjust decision threshold",
                    reason="Model biased to majority class",
                    expected_gain="+10–20% minority-class F1",
                    impact="HIGH",
                ))

        # Weak overall performance → embeddings
        if test_score < 0.70 and "leakage" not in label:
            out.append(Suggestion(
                priority=2, category="Model Upgrade",
                action="Move to word/sentence embeddings (install kaizenstat[nlp])",
                reason=f"Linear TF-IDF test score {test_score:.2f} is plateauing",
                expected_gain="Embeddings often add 5–15% on semantic tasks",
                impact="HIGH",
            ))
        # Low score → tuning
        if test_score < 0.80 and "leakage" not in label:
            out.append(Suggestion(
                priority=3, category="Tuning",
                action="Run train(tune=True) to jointly optimise TF-IDF + classifier",
                reason=f"Test score {test_score:.2f} has headroom",
                expected_gain="Typically +3–10%",
                impact="MEDIUM",
            ))
        return out

    def _from_data(self, df, target, text_col) -> List[Suggestion]:
        out = []
        n = len(df)

        # Small dataset → augmentation
        if n < 2000:
            out.append(Suggestion(
                priority=2, category="Data Volume",
                action="Augment text (back-translation, synonym swap) or collect more labelled data",
                reason=f"Only {n:,} documents — text models are data-hungry",
                expected_gain="More stable CV, better generalisation",
                impact="HIGH" if n < 500 else "MEDIUM",
            ))

        # Imbalance from raw labels
        if target and target in df.columns:
            y = df[target].dropna()
            if detect_task_type(y) == "classification":
                counts = y.value_counts(normalize=True)
                if len(counts) >= 2 and counts.iloc[-1] < 0.20:
                    out.append(Suggestion(
                        priority=1, category="Class Imbalance",
                        action="Apply class_weight='balanced' or SMOTE-style text oversampling",
                        reason=f"Minority class is {counts.iloc[-1]:.0%} of data",
                        expected_gain="+10–20% minority-class recall/F1",
                        impact="HIGH",
                    ))

        # Short documents → char n-grams
        if text_col and text_col in df.columns:
            avg_words = df[text_col].fillna("").astype(str).str.split().str.len().mean()
            if avg_words and avg_words < 5:
                out.append(Suggestion(
                    priority=2, category="Representation",
                    action="Add char n-grams — short docs lack enough word overlap",
                    reason=f"Average document is only {avg_words:.1f} words",
                    expected_gain="Better signal on short text",
                    impact="MEDIUM",
                ))
        return out


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_suggester = TextSuggester()


def suggest(
    df: pd.DataFrame, target: Optional[str] = None, text_col: Optional[str] = None,
    health_result=None, debug_result=None, validation_result=None,
) -> ImprovementReport:
    return _suggester.suggest(
        df, target, text_col=text_col,
        health_result=health_result, debug_result=debug_result,
        validation_result=validation_result,
    )
