"""Text Model Debugger — the NLP diagnosis engine (counterpart of debugger.py).

This is the core value of KaizenStat for NLP: it tells the user *why* their text
model is failing and *how* to fix it. Detects:

  • class-imbalance prediction bias   (model collapses to majority class)
  • sparse TF-IDF matrix              (> 95% zeros → weak representation)
  • weak feature representation       (tiny vocabulary / unigram-only)
  • overfitting to rare tokens        (large gap + hapax-heavy vocab)
  • underfitting due to simple model  (both scores low)

Reuses DebugResult / DebugIssue and the proven priority classifier from
ModelDebugger, so output is structurally identical to the tabular debugger.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from kaizenstat.debug.debugger import DebugIssue, DebugResult, ModelDebugger
from kaizenstat.utils.helpers import detect_task_type

warnings.filterwarnings("ignore")


class TextModelDebugger:
    """Priority-based diagnosis specialised for TF-IDF text pipelines."""

    def __init__(self) -> None:
        # Reuse the tabular classifier / health-score / reasoning machinery
        self._base = ModelDebugger()

    def model_failure(
        self,
        model: Any,
        X_train_text: pd.Series,
        X_test_text: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> DebugResult:
        task        = detect_task_type(y_test)
        train_score = model.score(X_train_text, y_train)
        test_score  = model.score(X_test_text, y_test)
        gap         = train_score - test_score
        avg_score   = (train_score + test_score) / 2

        label, severity, confidence = self._base._classify(train_score, test_score, gap)
        health_score = self._base._compute_health_score(label, gap, test_score)
        reasoning    = self._base._build_reasoning(label, train_score, test_score, gap, avg_score)

        # Text-specific structured issues + bullets
        vec_stats   = self._vectorizer_stats(model, X_train_text)
        issues: List[DebugIssue] = [DebugIssue(
            name=label.replace("_", " ").title(),
            description=f"Train={train_score:.3f}  Test={test_score:.3f}  Gap={gap:+.3f}",
            root_cause=reasoning,
            risk_level=severity,
            suggestion=self._primary_fix(label),
        )]
        issues += self._text_issues(model, X_train_text, X_test_text, y_train, y_test,
                                    task, gap, vec_stats)

        why_bullets = self._why_bullets(label, train_score, test_score, gap,
                                        task, y_test, vec_stats)

        from kaizenstat.debug.debugger import _LABEL_TO_DIAGNOSIS
        diagnosis = _LABEL_TO_DIAGNOSIS.get(label, "data_issue")

        result = DebugResult(
            task=task,
            train_score=round(train_score, 4),
            test_score=round(test_score, 4),
            gap=round(gap, 4),
            diagnosis=diagnosis,
            root_cause=reasoning,
            issues=issues,
            suggestions=self._suggestions(label, vec_stats),
            feature_importances=self._top_tokens(model),
            metrics={"train": round(train_score, 4), "test": round(test_score, 4)},
            avg_score=round(avg_score, 4),
            label=label,
            severity=severity,
            confidence=confidence,
            health_score=health_score,
            why_bullets=why_bullets,
        )
        result.display()
        return result

    # ------------------------------------------------------------------ #
    # Vectorizer introspection                                           #
    # ------------------------------------------------------------------ #

    def _get_vectorizer(self, model):
        if hasattr(model, "named_steps") and "tfidf" in model.named_steps:
            return model.named_steps["tfidf"]
        return None

    def _vectorizer_stats(self, model, X_train_text) -> Dict[str, Any]:
        vec = self._get_vectorizer(model)
        stats: Dict[str, Any] = {
            "vocab_size": 0, "sparsity": 0.0, "ngram_range": None,
            "hapax_ratio": 0.0, "avg_nonzero": 0.0,
        }
        if vec is None:
            return stats
        try:
            Xt = vec.transform(X_train_text.iloc[:5000] if len(X_train_text) > 5000 else X_train_text)
            stats["vocab_size"]  = len(getattr(vec, "vocabulary_", {}))
            stats["ngram_range"] = getattr(vec, "ngram_range", None)
            if sparse.issparse(Xt):
                total = Xt.shape[0] * Xt.shape[1]
                nz    = Xt.nnz
                stats["sparsity"]    = 1.0 - (nz / max(total, 1))
                stats["avg_nonzero"] = nz / max(Xt.shape[0], 1)
                # hapax: tokens appearing in only one document
                doc_freq = np.asarray((Xt > 0).sum(axis=0)).ravel()
                if doc_freq.size:
                    stats["hapax_ratio"] = float((doc_freq <= 1).mean())
        except Exception:
            pass
        return stats

    def _top_tokens(self, model) -> Optional[pd.Series]:
        vec = self._get_vectorizer(model)
        if vec is None:
            return None
        inner = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        # Unwrap calibrated/ensemble wrappers
        coef = getattr(inner, "coef_", None)
        if coef is None and hasattr(inner, "estimators_"):
            try:
                coefs = [e.coef_ for e in inner.estimators_ if hasattr(e, "coef_")]
                if coefs:
                    coef = np.mean(coefs, axis=0)
            except Exception:
                coef = None
        if coef is None and hasattr(inner, "calibrated_classifiers_"):
            try:
                base = inner.calibrated_classifiers_[0].estimator
                coef = getattr(base, "coef_", None)
            except Exception:
                coef = None
        if coef is None:
            return None
        try:
            coef = np.abs(np.asarray(coef))
            if coef.ndim > 1:
                coef = coef.mean(axis=0)
            names = vec.get_feature_names_out()
            if len(names) == len(coef):
                return pd.Series(coef, index=names).sort_values(ascending=False)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Text-specific issue detection                                      #
    # ------------------------------------------------------------------ #

    def _text_issues(self, model, X_tr, X_te, y_tr, y_te, task, gap, vec) -> List[DebugIssue]:
        issues: List[DebugIssue] = []

        # 1. Sparse TF-IDF matrix
        if vec["sparsity"] > 0.95:
            issues.append(DebugIssue(
                name="Sparse TF-IDF Matrix",
                description=f"{vec['sparsity']:.1%} of the TF-IDF matrix is zeros "
                            f"(avg {vec['avg_nonzero']:.0f} non-zero features/doc)",
                root_cause="Documents share few common tokens — representation is too sparse to learn from",
                risk_level="HIGH" if vec["sparsity"] > 0.99 else "MEDIUM",
                suggestion="Add character n-grams, reduce min_df, or move to embeddings",
            ))

        # 2. Weak feature representation
        if 0 < vec["vocab_size"] < 200 or vec["ngram_range"] == (1, 1):
            issues.append(DebugIssue(
                name="Weak Representation",
                description=f"Vocabulary={vec['vocab_size']:,}, ngram_range={vec['ngram_range']}",
                root_cause="Unigram-only / tiny vocabulary cannot capture phrases and context",
                risk_level="MEDIUM",
                suggestion="Use ngram_range=(1,2) or (1,3); consider word embeddings",
            ))

        # 3. Overfitting to rare tokens
        if gap > 0.10 and vec["hapax_ratio"] > 0.5:
            issues.append(DebugIssue(
                name="Rare-Token Overfitting",
                description=f"Gap={gap:+.3f} with {vec['hapax_ratio']:.0%} single-doc tokens",
                root_cause="Model is memorising rare/unique tokens that don't generalise",
                risk_level="HIGH",
                suggestion="Raise min_df to 3–5 to prune rare tokens; add regularisation",
            ))

        # 4. Class-imbalance prediction bias
        if task == "classification":
            counts = pd.Series(y_te).value_counts(normalize=True)
            if len(counts) > 1 and counts.iloc[-1] < 0.20:
                y_pred = model.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
                pred_classes = pd.Series(y_pred).nunique()
                if acc - f1 > 0.10 or pred_classes < len(counts):
                    issues.append(DebugIssue(
                        name="Imbalance Prediction Bias",
                        description=f"Acc {acc:.3f} vs F1 {f1:.3f}; "
                                    f"predicts {pred_classes}/{len(counts)} classes",
                        root_cause="Model collapses toward the majority class on imbalanced text",
                        risk_level="HIGH",
                        suggestion="Use class_weight='balanced' (default here), oversample, or adjust threshold",
                    ))
        return issues

    # ------------------------------------------------------------------ #
    # Narrative                                                           #
    # ------------------------------------------------------------------ #

    def _why_bullets(self, label, train, test, gap, task, y_test, vec) -> List[str]:
        bullets: List[str] = []
        if "overfitting" in label:
            bullets.append(f"• Train-test gap is {gap:.2f} — model memorised training text")
            if vec["hapax_ratio"] > 0.5:
                bullets.append(f"• {vec['hapax_ratio']:.0%} of vocabulary appears in only one document — rare-token memorisation")
        elif "underfitting" in label:
            bullets.append(f"• Both train ({train:.2f}) and test ({test:.2f}) are low — representation too weak")
            if vec["ngram_range"] == (1, 1):
                bullets.append("• Unigram-only TF-IDF — phrases and word order are lost")
        elif label in ("data_leakage", "leakage_risk"):
            bullets.append("• Near-perfect scores — a giveaway token likely encodes the label directly")

        if vec["sparsity"] > 0.95:
            bullets.append(f"• TF-IDF matrix is {vec['sparsity']:.0%} zeros — documents barely overlap in vocabulary")
        if 0 < vec["vocab_size"] < 200:
            bullets.append(f"• Tiny vocabulary ({vec['vocab_size']} terms) — not enough lexical signal")

        if task == "classification":
            counts = pd.Series(y_test).value_counts(normalize=True)
            if len(counts) >= 2 and counts.iloc[-1] < 0.20:
                bullets.append(f"• Class imbalance: minority is {counts.iloc[-1]:.0%} — bias toward majority class")
        return bullets

    def _primary_fix(self, label: str) -> str:
        from kaizenstat.debug.debugger import _FIX_SUGGESTIONS
        return _FIX_SUGGESTIONS.get(label, ["Use cross-validation"])[0]

    def _suggestions(self, label: str, vec: Dict[str, Any]) -> List[str]:
        sugg: List[str] = []
        if vec["sparsity"] > 0.95:
            sugg.append("Reduce min_df and add char n-grams (analyzer='char_wb', ngram_range=(2,5))")
        if vec["ngram_range"] == (1, 1):
            sugg.append("Switch to word ngram_range=(1,2) to capture bigrams")
        if vec["hapax_ratio"] > 0.5:
            sugg.append("Set min_df=3 to prune one-off tokens")
        sugg.append("Run train(tune=True) to optimise the TF-IDF + model jointly")
        sugg.append("If performance plateaus, move to embeddings (kaizenstat[nlp])")
        return sugg


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_debugger = TextModelDebugger()


def model_failure(model, X_train_text, X_test_text, y_train, y_test) -> DebugResult:
    return _debugger.model_failure(model, X_train_text, X_test_text, y_train, y_test)
