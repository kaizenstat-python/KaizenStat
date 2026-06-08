"""Text Data Health Scoring Engine — NLP counterpart of scorer.py.

Reuses HealthResult / HealthPenalty so DataDoctor returns the same structured
output regardless of whether the dataset is tabular or text.
"""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from kaizenstat.health.scorer import HealthPenalty, HealthResult
from kaizenstat.utils.helpers import (
    dominant_text_column,
    score_to_grade,
    validate_dataframe,
)

_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL_RE  = re.compile(r"[^a-zA-Z0-9\s]")
_HTML_RE     = re.compile(r"<[^>]+>")


class TextHealthScorer:
    """Compute a 0–100 health score for a free-text column."""

    _LIMITS = {
        "empty_short":  25.0,
        "duplicates":   20.0,
        "noise":        20.0,
        "vocab":        15.0,
        "length_var":   10.0,
        "imbalance":    20.0,
    }

    def score(self, df: pd.DataFrame, target: Optional[str] = None,
              text_col: Optional[str] = None) -> float:
        return self.breakdown(df, target, text_col).score

    def report(self, df: pd.DataFrame, target: Optional[str] = None,
               text_col: Optional[str] = None) -> HealthResult:
        result = self.breakdown(df, target, text_col)
        result.display()
        return result

    def breakdown(self, df: pd.DataFrame, target: Optional[str] = None,
                  text_col: Optional[str] = None) -> HealthResult:
        validate_dataframe(df, target)
        text_col = text_col or dominant_text_column(df, exclude=[target] if target else None)
        if text_col is None:
            raise ValueError("No dominant text column found — use the tabular HealthScorer instead.")

        s = df[text_col].fillna("").astype(str)
        penalties: List[HealthPenalty] = []
        total = 0.0

        total += self._empty_short(s, penalties)
        total += self._duplicates(s, penalties)
        total += self._noise(s, penalties)
        total += self._vocab(s, penalties)
        total += self._length_variance(s, penalties)
        total += self._imbalance(df, target, penalties)

        score = round(max(0.0, 100.0 - total), 1)
        risk = (
            "CRITICAL" if score < 40 else
            "HIGH"     if score < 60 else
            "MEDIUM"   if score < 80 else
            "LOW"
        )
        return HealthResult(
            score=score,
            grade=score_to_grade(score),
            risk_level=risk,
            penalties=penalties,
            summary=self._summary(score, text_col),
            rows=len(df),
            columns=len(df.columns),
        )

    # ------------------------------------------------------------------ #
    # Penalty calculators                                                 #
    # ------------------------------------------------------------------ #

    def _empty_short(self, s, out):
        words = s.str.split().str.len().fillna(0)
        empty_short = ((words <= 2)).mean()
        if empty_short < 0.02:
            return 0.0
        deduct = min(self._LIMITS["empty_short"], empty_short * 60)
        risk = "HIGH" if empty_short > 0.20 else "MEDIUM" if empty_short > 0.05 else "LOW"
        out.append(HealthPenalty(
            name="Empty / Short Text",
            penalty=-round(deduct, 1),
            reason=f"{empty_short:.0%} of rows have ≤ 2 words — too little signal",
            risk_level=risk,
        ))
        return deduct

    def _duplicates(self, s, out):
        norm = s.str.strip().str.lower()
        non_empty = norm[norm.str.len() > 0]
        if non_empty.empty:
            return 0.0
        dup = non_empty.duplicated().mean()
        if dup < 0.02:
            return 0.0
        deduct = min(self._LIMITS["duplicates"], dup * 50)
        risk = "HIGH" if dup > 0.30 else "MEDIUM"
        out.append(HealthPenalty(
            name="Duplicate Text",
            penalty=-round(deduct, 1),
            reason=f"{dup:.0%} of documents are exact duplicates",
            risk_level=risk,
        ))
        return deduct

    def _noise(self, s, out):
        sample = s.sample(min(len(s), 1000), random_state=42)
        def noise_ratio(t: str) -> float:
            if not t:
                return 0.0
            specials = len(_SPECIAL_RE.findall(t))
            return specials / max(len(t), 1)
        has_url  = sample.str.contains(_URL_RE).mean()
        has_html = sample.str.contains(_HTML_RE).mean()
        avg_noise = sample.apply(noise_ratio).mean()
        triggers = []
        if has_url > 0.05:
            triggers.append(f"{has_url:.0%} contain URLs")
        if has_html > 0.05:
            triggers.append(f"{has_html:.0%} contain HTML tags")
        if avg_noise > 0.15:
            triggers.append(f"{avg_noise:.0%} special-char density")
        if not triggers:
            return 0.0
        deduct = min(self._LIMITS["noise"], avg_noise * 60 + has_url * 30 + has_html * 30)
        out.append(HealthPenalty(
            name="Text Noise",
            penalty=-round(deduct, 1),
            reason="; ".join(triggers) + " — clean before vectorising",
            risk_level="MEDIUM",
        ))
        return deduct

    def _vocab(self, s, out):
        sample = s.sample(min(len(s), 2000), random_state=42)
        tokens = sample.str.lower().str.split().explode().dropna()
        if tokens.empty:
            return 0.0
        vocab = tokens.nunique()
        ttr = vocab / max(len(tokens), 1)   # type-token ratio
        if ttr > 0.05 and vocab > 50:
            return 0.0
        deduct = min(self._LIMITS["vocab"], (0.05 - ttr) * 200 if ttr < 0.05 else 5)
        out.append(HealthPenalty(
            name="Low Vocabulary",
            penalty=-round(max(deduct, 0), 1),
            reason=f"Vocabulary={vocab:,}, type-token ratio={ttr:.3f} — limited lexical diversity",
            risk_level="MEDIUM" if vocab < 50 else "LOW",
        ))
        return max(deduct, 0)

    def _length_variance(self, s, out):
        words = s.str.split().str.len().fillna(0)
        if words.mean() == 0:
            return 0.0
        cv = words.std() / (words.mean() + 1e-9)
        if cv < 1.5:
            return 0.0
        deduct = min(self._LIMITS["length_var"], (cv - 1.5) * 8)
        out.append(HealthPenalty(
            name="Length Variance",
            penalty=-round(deduct, 1),
            reason=f"Document length varies wildly (CV={cv:.2f}) — mix of fragments and long docs",
            risk_level="LOW",
        ))
        return deduct

    def _imbalance(self, df, target, out):
        if not target or target not in df.columns:
            return 0.0
        y = df[target].dropna()
        if y.nunique() > 50:
            return 0.0
        counts = y.value_counts(normalize=True)
        if len(counts) <= 1:
            return 0.0
        minority = counts.iloc[-1]
        if minority >= 0.10:
            return 0.0
        deduct = min(self._LIMITS["imbalance"], (0.10 - minority) * 200)
        out.append(HealthPenalty(
            name="Class Imbalance",
            penalty=-round(deduct, 1),
            reason=f"Minority label '{counts.index[-1]}' = {minority:.1%} — text models bias to majority",
            risk_level="HIGH" if minority < 0.05 else "MEDIUM",
        ))
        return deduct

    @staticmethod
    def _summary(score: float, col: str) -> str:
        head = f"Text column '{col}': "
        if score >= 90:
            return head + "Excellent text quality. Ready for vectorisation."
        if score >= 75:
            return head + "Good quality. Minor cleaning recommended."
        if score >= 60:
            return head + "Moderate noise/sparsity. Clean before training."
        if score >= 40:
            return head + "Significant text-quality problems detected."
        return head + "Critical state. Heavy preprocessing required."


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_scorer = TextHealthScorer()


def score(df: pd.DataFrame, target: Optional[str] = None, text_col: Optional[str] = None) -> float:
    return _scorer.score(df, target, text_col)


def report(df: pd.DataFrame, target: Optional[str] = None, text_col: Optional[str] = None) -> HealthResult:
    return _scorer.report(df, target, text_col)


def breakdown(df: pd.DataFrame, target: Optional[str] = None, text_col: Optional[str] = None) -> HealthResult:
    return _scorer.breakdown(df, target, text_col)
