"""Text validation engine — NLP counterpart of checker.py.

Reuses ValidationReport / ValidationIssue for consistent structured output.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import pandas as pd
from rich.console import Console

from kaizenstat.validate.checker import ValidationIssue, ValidationReport
from kaizenstat.utils.helpers import dominant_text_column, validate_dataframe

console = Console()

# Minimal built-in stopword list — avoids an nltk dependency.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "as", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "my", "your", "our", "me", "him", "us", "so", "not", "no", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "about", "into", "than", "too", "very", "just", "out", "up", "down",
}


class TextValidator:
    """Validates text-data assumptions before NLP modelling."""

    def assumptions(self, df: pd.DataFrame, target: Optional[str] = None,
                    text_col: Optional[str] = None) -> ValidationReport:
        validate_dataframe(df, target)
        text_col = text_col or dominant_text_column(df, exclude=[target] if target else None)
        if text_col is None:
            raise ValueError("No dominant text column found.")

        s = df[text_col].fillna("").astype(str)
        tokens = self._tokenize(s)

        issues: List[ValidationIssue] = []
        issues += self._check_token_skew(tokens, text_col)
        issues += self._check_stopword_dominance(tokens, text_col)
        issues += self._check_rare_explosion(tokens, text_col)
        if target and target in df.columns:
            issues += self._check_label_leakage(df, target, text_col)

        report = ValidationReport(
            passed=not any(i.risk_level == "HIGH" for i in issues),
            issues=issues,
            checks_run=4 if target else 3,
        )
        return report

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(s: pd.Series, sample: int = 3000) -> List[str]:
        if len(s) > sample:
            s = s.sample(sample, random_state=42)
        toks = s.str.lower().str.findall(r"[a-z]{2,}")
        return [t for row in toks for t in row]

    # ------------------------------------------------------------------ #
    # Checks                                                              #
    # ------------------------------------------------------------------ #

    def _check_token_skew(self, tokens, col):
        if not tokens:
            return []
        counts = Counter(tokens)
        total = sum(counts.values())
        top10 = sum(c for _, c in counts.most_common(10))
        share = top10 / max(total, 1)
        if share <= 0.50:
            return []
        top_words = ", ".join(w for w, _ in counts.most_common(5))
        return [ValidationIssue(
            check="Token Frequency Skew",
            issue=f"Top 10 tokens make up {share:.0%} of all tokens",
            reason="A few tokens dominate the corpus → TF-IDF may under-weight rare signal",
            risk_level="MEDIUM",
            recommendation="Add sublinear_tf=True and tune max_df to down-weight frequent tokens",
            column=f"{col}: {top_words}",
        )]

    def _check_stopword_dominance(self, tokens, col):
        if not tokens:
            return []
        stop = sum(1 for t in tokens if t in _STOPWORDS)
        ratio = stop / max(len(tokens), 1)
        if ratio <= 0.55:
            return []
        return [ValidationIssue(
            check="Stopword Dominance",
            issue=f"{ratio:.0%} of tokens are stopwords",
            reason="High stopword ratio dilutes discriminative features",
            risk_level="MEDIUM" if ratio > 0.70 else "LOW",
            recommendation="Pass stop_words='english' to the TF-IDF vectoriser",
            column=col,
        )]

    def _check_rare_explosion(self, tokens, col):
        if not tokens:
            return []
        counts = Counter(tokens)
        vocab = len(counts)
        hapax = sum(1 for _, c in counts.items() if c == 1)
        ratio = hapax / max(vocab, 1)
        if ratio <= 0.60 or vocab < 100:
            return []
        return [ValidationIssue(
            check="Rare Token Explosion",
            issue=f"{ratio:.0%} of the {vocab:,}-word vocabulary appears only once",
            reason="Hapax-heavy vocabulary explodes feature space and causes overfitting",
            risk_level="HIGH" if ratio > 0.75 else "MEDIUM",
            recommendation="Set min_df=2 (or higher) to prune one-off tokens",
            column=col,
        )]

    def _check_label_leakage(self, df, target, col):
        y = df[target].fillna("__nan__").astype(str)
        s = df[col].fillna("").astype(str).str.lower()
        # Build token presence per class; flag tokens that appear almost exclusively
        # in one class AND are common enough to matter.
        sample_idx = s.sample(min(len(s), 3000), random_state=42).index
        s = s.loc[sample_idx]
        y = y.loc[sample_idx]

        token_class: dict = {}
        token_total: dict = {}
        for txt, lbl in zip(s, y):
            seen = set(re.findall(r"[a-z]{3,}", txt))
            for tok in seen:
                token_total[tok] = token_total.get(tok, 0) + 1
                token_class.setdefault(tok, Counter())[lbl] += 1

        n = len(s)
        suspects = []
        for tok, tot in token_total.items():
            if tot < max(5, n * 0.01):       # must be reasonably frequent
                continue
            dominant = token_class[tok].most_common(1)[0][1]
            purity = dominant / tot
            if purity >= 0.98 and tot >= n * 0.02:
                suspects.append(f"{tok}({purity:.0%})")

        if not suspects:
            return []
        console.print(
            f"[bold red]🚨 Possible text label leakage: {suspects[:5]}[/bold red]\n"
            f"   These tokens appear almost exclusively in one class — "
            f"the model may be memorising giveaway words instead of learning language."
        )
        return [ValidationIssue(
            check="Label Leakage (text)",
            issue=f"{len(suspects)} token(s) almost perfectly predict the label",
            reason="Tokens >98% concentrated in one class → inflated, non-generalisable scores",
            risk_level="HIGH",
            recommendation="Investigate/remove giveaway tokens; verify they are not metadata",
            column=", ".join(suspects[:3]),
        )]


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_validator = TextValidator()


def assumptions(df: pd.DataFrame, target: Optional[str] = None,
                text_col: Optional[str] = None) -> ValidationReport:
    return _validator.assumptions(df, target, text_col)
