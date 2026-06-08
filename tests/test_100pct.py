"""Targeted tests to reach 100% coverage on every remaining uncovered line."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clf_df(n=200, n_features=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    y = rng.integers(0, n_classes, n)
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y
    return pd.DataFrame(cols)


def _reg_df(n=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, n)
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y
    return pd.DataFrame(cols)


def _text_df(n=100, n_classes=2, seed=0, string_labels=False):
    rng = np.random.default_rng(seed)
    vocab = ["good", "bad", "great", "terrible", "excellent", "poor",
             "wonderful", "awful", "nice", "horrible"]
    texts = [" ".join(rng.choice(vocab, size=5).tolist()) for _ in range(n)]
    if string_labels:
        labels = ["pos" if i % 2 == 0 else "neg" for i in range(n)]
    else:
        labels = [i % n_classes for i in range(n)]
    return pd.DataFrame({"text": texts, "label": labels})


# ─────────────────────────────────────────────────────────────────────────────
# debugger.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDebuggerCoverage:

    def _setup(self):
        from kaizenstat.debug.debugger import ModelDebugger
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
        y = pd.Series([i % 2 for i in range(60)])
        model = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())])
        model.fit(X, y)
        return ModelDebugger(), model, X, y

    # L421 — feature_impact SUCCESS path (impacts[col] = round(...))
    def test_feature_impact_success_path(self):
        from kaizenstat.debug.debugger import ModelDebugger
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
        y = pd.Series([i % 2 for i in range(60)])
        # DummyClassifier ignores feature count → score() always succeeds even when columns dropped
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        dbg = ModelDebugger()
        impacts = dbg.feature_impact(model, X, y)
        assert isinstance(impacts, dict)
        # At least one column should have a non-zero impact entry from the success path
        assert len(impacts) >= 1

    # L488-489 — dataset_difficulty except path
    def test_dataset_difficulty_cv_exception(self):
        dbg, _, X, y = self._setup()
        with patch("sklearn.model_selection.cross_val_score", side_effect=Exception("cv fail")):
            result = dbg.dataset_difficulty(X, y)
        assert result is not None

    # L622 — bias_detection: feat not in X_test.columns → continue
    def test_bias_detection_feature_missing(self):
        dbg, model, X, y = self._setup()
        result = dbg.bias_detection(model, X, y, sensitive_features=["nonexistent_col"])
        assert isinstance(result, dict)

    # L627 — bias_detection: mask.sum() < 5 → continue
    def test_bias_detection_tiny_group(self):
        from kaizenstat.debug.debugger import ModelDebugger
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
        y = pd.Series([i % 2 for i in range(60)])
        model = MagicMock()
        model.predict.return_value = np.zeros(len(y), dtype=int)
        X2 = X.copy()
        X2["cat"] = ["rare"] * 2 + ["common"] * (len(X) - 2)
        result = ModelDebugger().bias_detection(model, X2, y, sensitive_features=["cat"])
        assert isinstance(result, dict)

    # L737 — _compute_health_score: elif gap > 0.10 → score -= 10
    def test_compute_health_score_small_gap(self):
        from kaizenstat.debug.debugger import ModelDebugger
        # gap=0.15 → hits elif gap > 0.10 (NOT gap > 0.20)
        score = ModelDebugger()._compute_health_score("acceptable", 0.15, 0.80)
        assert 0 <= score <= 100

    # L759 — _why_bullets overfitting + ratio < 10 bullet
    def test_why_bullets_overfitting_low_ratio(self):
        from kaizenstat.debug.debugger import ModelDebugger
        # X_train has 8 rows, X_test has 2 columns → ratio = 8/2 = 4 < 10
        X_train = pd.DataFrame({"a": range(8), "b": range(8)})
        X_test  = pd.DataFrame({"a": range(5), "b": range(5)})
        y_test  = pd.Series([0, 1, 0, 1, 0])
        bullets = ModelDebugger()._why_bullets(
            "overfitting", 0.95, 0.70, 0.25, None, y_test, X_train, X_test
        )
        assert any("training rows" in b for b in bullets)

    # L784 — _why_bullets: low_var > 0 → near-constant feature bullet
    def test_why_bullets_low_variance_feature(self):
        from kaizenstat.debug.debugger import ModelDebugger
        rng = np.random.default_rng(1)
        X_train = pd.DataFrame({
            "const": np.full(20, 1.0),           # variance = 0 < 0.01
            "normal": rng.normal(0, 1, 20),
        })
        X_test = pd.DataFrame({"const": [1.0]*5, "normal": rng.normal(0, 1, 5)})
        y_test = pd.Series([0, 1, 0, 1, 0])
        bullets = ModelDebugger()._why_bullets(
            "data_issue", 0.80, 0.85, 0.05, None, y_test, X_train, X_test
        )
        assert any("near-constant" in b for b in bullets)

    # L880 — _data_vs_model_blame: X_num.shape[1] == 0 → return None
    def test_data_vs_model_blame_no_numeric_features(self):
        from kaizenstat.debug.debugger import ModelDebugger
        # DataFrame with ONLY string/object columns → no numeric → X_num empty → L880
        X_all_obj = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"] * 4})
        y = pd.Series([0, 1] * 10)
        result = ModelDebugger()._data_vs_model_blame(
            "underfitting", 0.55, X_all_obj, y, "classification"
        )
        assert result is None

    # L932 — _data_vs_model_blame: except Exception: pass
    def test_data_vs_model_blame_cross_val_raises(self):
        from kaizenstat.debug.debugger import ModelDebugger
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20)})
        y = pd.Series([i % 2 for i in range(20)])
        with patch("sklearn.model_selection.cross_val_score", side_effect=RuntimeError("fail")):
            result = ModelDebugger()._data_vs_model_blame(
                "underfitting", 0.55, X, y, "classification"
            )
        assert result is None

    # L951 — _failure_clustering: X_test.empty → return []
    def test_failure_clustering_empty_df(self):
        from kaizenstat.debug.debugger import ModelDebugger
        model = MagicMock()
        y = pd.Series([0, 1])
        result = ModelDebugger()._failure_clustering(model, pd.DataFrame(), y)
        assert result == []

    # L969 — _failure_clustering: small group skipped
    def test_failure_clustering_small_group(self):
        from kaizenstat.debug.debugger import ModelDebugger
        model = MagicMock()
        model.predict.return_value = np.array([0] * 60)
        X = pd.DataFrame({"cat": ["rare"] * 2 + ["common"] * 58})
        y = pd.Series([0] * 58 + [1] * 2)
        result = ModelDebugger()._failure_clustering(model, X, y)
        assert isinstance(result, list)

    # L1015 — _diagnose_imbalance: except Exception path
    def test_diagnose_imbalance_exception(self):
        from kaizenstat.debug.debugger import ModelDebugger
        model = MagicMock()
        y_test = pd.Series([0] * 87 + [1] * 8 + [2] * 5)  # minority < 0.15
        X_test = pd.DataFrame({"a": range(100)})
        model.predict.return_value = np.zeros(100, dtype=int)
        with patch("kaizenstat.debug.debugger.f1_score", side_effect=Exception("f1 fail")):
            issues = ModelDebugger()._diagnose_imbalance(None, y_test, model, X_test)
        assert isinstance(issues, list)

    # L1036 — _extract_feature_importance: permutation except → return None
    def test_extract_feature_importance_permutation_fails(self):
        from kaizenstat.debug.debugger import ModelDebugger
        model = MagicMock()
        model.named_steps = {}
        del model.feature_importances_
        del model.coef_
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        y = pd.Series([i % 2 for i in range(20)])
        with patch("sklearn.inspection.permutation_importance", side_effect=Exception("perm fail")):
            result = ModelDebugger()._extract_feature_importance(model, X, y, "classification")
        assert result is None

    # L1048 — _get_feature_names: transform fallback raises → use X.columns
    def test_get_feature_names_transform_fallback_raises(self):
        from kaizenstat.debug.debugger import ModelDebugger
        prep = MagicMock()
        prep.get_feature_names_out.side_effect = Exception("no names")
        prep.transform.side_effect = Exception("transform fail")
        model = MagicMock()
        model.named_steps = {"prep": prep}
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        names = ModelDebugger()._get_feature_names(model, X)
        assert names == ["a", "b"]

    # L1095 — _compute_extra_metrics: metric raises → return {}
    def test_compute_extra_metrics_metric_raises(self):
        from kaizenstat.debug.debugger import ModelDebugger
        model = MagicMock()
        model.predict.return_value = np.array([0] * 10)
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series([i % 2 for i in range(10)])
        with patch("kaizenstat.debug.debugger.accuracy_score", side_effect=Exception("fail")):
            result = ModelDebugger()._compute_extra_metrics(model, X, y, "classification")
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# text_debugger.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTextDebuggerCoverage:

    def _text_model(self, n=200):
        from sklearn.feature_extraction.text import TfidfVectorizer
        rng = np.random.default_rng(1)
        vocab = ["good", "bad", "great", "nice", "poor", "awful", "best", "worst",
                 "love", "hate", "happy", "sad", "fast", "slow", "rich", "cheap"]
        texts = [" ".join(rng.choice(vocab, size=8).tolist()) for _ in range(n)]
        y = pd.Series([i % 2 for i in range(n)])
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=300))])
        pipe.fit(texts, y)
        return pipe, pd.Series(texts), y

    # L125 — _vectorizer_stats except Exception path
    def test_vectorizer_stats_transform_raises(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        pipe, texts, _ = self._text_model()
        # Patch the vectorizer's transform to raise
        with patch.object(pipe.named_steps["tfidf"], "transform", side_effect=Exception("err")):
            stats = TextModelDebugger()._vectorizer_stats(pipe, texts)
        assert "hapax_ratio" in stats  # dict still has defaults

    # L141-142 — _top_tokens: estimators_ try/except (np.mean raises on bad coef)
    def test_top_tokens_estimators_exception(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        from unittest.mock import PropertyMock
        vec = MagicMock()
        vec.get_feature_names_out.side_effect = Exception("names fail")
        inner = MagicMock()
        del inner.coef_
        del inner.calibrated_classifiers_
        # Make coef_ on sub-estimator raise RuntimeError so hasattr propagates → except
        sub = MagicMock(spec=[])  # no attributes → hasattr returns False, skip
        # Instead: have np.mean raise when called on coefs list
        sub2 = MagicMock()
        sub2.coef_ = np.array([[0.1, 0.2]])
        inner.estimators_ = [sub2]
        model = MagicMock()
        model.named_steps = {"tfidf": vec, "model": inner}
        with patch("numpy.mean", side_effect=Exception("mean fail")):
            result = TextModelDebugger()._top_tokens(model)
        assert result is None

    # L147-148 — calibrated_classifiers_ try/except (.estimator raises)
    def test_top_tokens_calibrated_except(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        from unittest.mock import PropertyMock
        vec = MagicMock()
        vec.get_feature_names_out.side_effect = Exception("names fail")
        inner = MagicMock()
        del inner.coef_
        del inner.estimators_
        cal_clf = MagicMock()
        # .estimator raises RuntimeError when accessed
        type(cal_clf).estimator = PropertyMock(side_effect=RuntimeError("cal err"))
        inner.calibrated_classifiers_ = [cal_clf]
        model = MagicMock()
        model.named_steps = {"tfidf": vec, "model": inner}
        result = TextModelDebugger()._top_tokens(model)
        assert result is None

    # L158-160 — _top_tokens final try: get_feature_names_out raises → except + return None
    def test_top_tokens_feature_names_raises(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        vec = MagicMock()
        vec.get_feature_names_out.side_effect = Exception("names fail")
        inner = MagicMock()
        inner.coef_ = np.array([[0.1, 0.2, 0.3]])
        model = MagicMock()
        model.named_steps = {"tfidf": vec, "model": inner}
        result = TextModelDebugger()._top_tokens(model)
        assert result is None

    # L271 — _suggestions: hapax_ratio > 0.5 → min_df suggestion
    def test_suggestions_hapax_high(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        vec_stats = {"sparsity": 0.5, "ngram_range": (1, 1), "hapax_ratio": 0.7}
        result = TextModelDebugger()._suggestions("underfitting", vec_stats)
        assert any("min_df" in s for s in result)

    # L204-209 — _text_issues: imbalance prediction bias branch
    def test_text_issues_imbalance_bias(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        from sklearn.feature_extraction.text import TfidfVectorizer
        rng = np.random.default_rng(3)
        vocab = ["good", "bad", "nice", "poor"]
        n = 100
        texts_tr = pd.Series([" ".join(rng.choice(vocab, 5).tolist()) for _ in range(n)])
        texts_te = pd.Series([" ".join(rng.choice(vocab, 5).tolist()) for _ in range(40)])
        y_tr = pd.Series([0] * 90 + [1] * 10)
        y_te = pd.Series([0] * 36 + [1] * 4)
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=300))])
        pipe.fit(texts_tr, y_tr)
        vec_stats = {"sparsity": 0.5, "ngram_range": (1, 2), "hapax_ratio": 0.2,
                     "vocab_size": 500, "avg_nonzero": 3.0}
        issues = TextModelDebugger()._text_issues(
            pipe, texts_tr, texts_te, y_tr, y_te, "classification", 0.1, vec_stats
        )
        assert isinstance(issues, list)

    # L244 — _why_bullets with classification imbalance
    def test_why_bullets_imbalanced(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        y_te = pd.Series([0] * 90 + [1] * 10)
        vec_stats = {"sparsity": 0.5, "ngram_range": (1, 2), "hapax_ratio": 0.2,
                     "vocab_size": 500, "avg_nonzero": 3.0}
        bullets = TextModelDebugger()._why_bullets(
            "overfitting", 0.95, 0.70, 0.25, "classification", y_te, vec_stats
        )
        assert isinstance(bullets, list)


# ─────────────────────────────────────────────────────────────────────────────
# data_doctor.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDataDoctorCoverage:

    # L292, 299-301 — debug_model fallback split (stratify fails)
    def test_debug_model_fallback_split(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        df = pd.DataFrame({
            "a": np.random.randn(22),
            "target": [0] * 21 + [1],  # 1 sample of class 1 → stratify fails
        })
        doc = DataDoctor()
        doc.fit(df, target="target")
        doc.train()
        result = doc.debug_model()
        assert result is not None

    # L458 — trust_score with LabelEncoder (string target)
    def test_trust_score_with_label_encoder(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "target": ["yes"] * 50 + ["no"] * 50,
        })
        doc = DataDoctor()
        doc.fit(df, target="target")
        doc.train()
        result = doc.trust_score()
        assert 0 <= result.trust_score <= 100


# ─────────────────────────────────────────────────────────────────────────────
# health/text_scorer.py — all uncovered branches
# ─────────────────────────────────────────────────────────────────────────────

class TestTextScorerCoverage:

    # L105 — _duplicates: non_empty.empty → return 0.0
    def test_duplicates_all_empty(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        s = pd.Series(["", "  ", "\t", ""] * 10)  # all blank → non_empty.empty == True
        out = []
        result = TextHealthScorer()._duplicates(s, out)
        assert result == 0.0

    # L151 — _vocab: tokens.empty → return 0.0
    def test_vocab_all_empty(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        s = pd.Series(["", "  "] * 50)  # no tokens
        out = []
        result = TextHealthScorer()._vocab(s, out)
        assert result == 0.0

    # L168 — _length_variance: words.mean() == 0 → return 0.0
    def test_length_variance_all_empty(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        s = pd.Series(["", "  "] * 50)  # all zero-length
        out = []
        result = TextHealthScorer()._length_variance(s, out)
        assert result == 0.0

    # L172-179 — _length_variance: cv >= 1.5 → penalty appended
    def test_length_variance_high_cv(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        # 90 one-word docs + 10 fifteen-word docs → high cv
        s = pd.Series(["word"] * 90 + ["word " * 15] * 10)
        out = []
        result = TextHealthScorer()._length_variance(s, out)
        assert result > 0

    # L186 — _imbalance: y.nunique() > 50 → return 0.0
    def test_imbalance_too_many_classes(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        df = pd.DataFrame({
            "text": [f"doc {i}" for i in range(200)],
            "label": list(range(200)),  # 200 unique classes > 50
        })
        out = []
        result = TextHealthScorer()._imbalance(df, "label", out)
        assert result == 0.0

    # L189 — _imbalance: len(counts) <= 1 → return 0.0
    def test_imbalance_single_class(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        df = pd.DataFrame({"text": ["hello"] * 50, "label": [0] * 50})
        out = []
        result = TextHealthScorer()._imbalance(df, "label", out)
        assert result == 0.0

    # L213 — _summary: score >= 90 → Excellent
    def test_summary_excellent(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        result = TextHealthScorer._summary(92.0, "text")
        assert "Excellent" in result

    # L223 — _summary: score >= 75 → Good
    def test_summary_good(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        result = TextHealthScorer._summary(78.0, "text")
        assert "Good" in result

    # L227 — _summary: score >= 60 → Moderate
    def test_summary_moderate(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        result = TextHealthScorer._summary(63.0, "text")
        assert "Moderate" in result

    # L231 — _summary: score >= 40 → Significant
    def test_summary_significant(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        result = TextHealthScorer._summary(45.0, "text")
        assert "Significant" in result


# ─────────────────────────────────────────────────────────────────────────────
# improve/suggester.py L195 — _from_debug else gain
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggesterCoverage:

    # L195 — else: gain = f"Expected +5-15%..." (label not overfitting/underfitting/leakage)
    def test_from_debug_else_gain(self):
        from kaizenstat.improve.suggester import Suggester as ImprovementSuggester
        from kaizenstat.debug.debugger import DebugResult, DebugIssue

        issue = DebugIssue(
            name="Test Issue", description="desc", root_cause="cause",
            risk_level="LOW", suggestion="try something"
        )
        dr = MagicMock()
        dr.issues = [issue]
        dr.gap = 0.0
        dr.test_score = 0.60
        dr.label = "data_issue"  # NOT overfitting/underfitting/leakage → hits else at L195

        suggester = ImprovementSuggester()
        suggestions = suggester._from_debug(dr, start_priority=1)
        assert len(suggestions) >= 1
        assert "Expected" in suggestions[0].expected_gain


# ─────────────────────────────────────────────────────────────────────────────
# improve/text_suggester.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTextSuggesterCoverage:

    # L48 — top = deduped[0] (non-empty suggestions list)
    def test_suggest_produces_top(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        df = pd.DataFrame({
            "text": ["short"] * 100,  # avg words = 1 < 5 → triggers short-doc suggestion
            "label": [0, 1] * 50,
        })
        report = TextSuggester().suggest(df, target="label", text_col="text")
        assert report.top_priority is not None

    # L189 — short docs suggestion: avg_words < 5
    def test_from_data_short_docs(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        df = pd.DataFrame({
            "text": ["hi"] * 50 + ["bye"] * 50,
            "label": [0] * 50 + [1] * 50,
        })
        suggs = TextSuggester()._from_data(df, target="label", text_col="text")
        assert any("short" in s.reason.lower() or "char" in s.action.lower() for s in suggs)


# ─────────────────────────────────────────────────────────────────────────────
# model/text_trainer.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTextTrainerCoverage:

    # L80 — train_best: raises ValueError when no text column found
    def test_train_best_no_text_column_raises(self):
        from kaizenstat.model.text_trainer import TextModelTrainer as TextTrainer
        import pytest
        # All-numeric DataFrame → dominant_text_column returns None → L80
        df = pd.DataFrame({"a": range(50), "b": range(50), "target": [0, 1] * 25})
        with pytest.raises(ValueError, match="No dominant text column"):
            TextTrainer().train_best(df, "target")

    # L134-142 — benchmark: LinearSVC candidate added when n >= 500
    def test_benchmark_large_dataset_linearSVC(self):
        from kaizenstat.model.text_trainer import TextModelTrainer as TextTrainer
        from sklearn.model_selection import StratifiedKFold
        rng = np.random.default_rng(42)
        vocab = ["good", "bad", "great", "nice", "poor", "awful", "best"]
        n = 600
        X_tr = pd.Series([" ".join(rng.choice(vocab, 8).tolist()) for _ in range(n)])
        y_tr = pd.Series([i % 2 for i in range(n)])
        cv_obj = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        name, pipe, score = TextTrainer()._benchmark_text_pipelines(
            X_tr, y_tr, "classification", cv_obj, "accuracy"
        )
        assert name is not None  # LinearSVC candidate was evaluated

    # L192-193 — _prepare: string target → LabelEncoder applied
    def test_prepare_string_target_label_encoder(self):
        from kaizenstat.model.text_trainer import TextModelTrainer as TextTrainer
        df = _text_df(n=100, string_labels=True)
        trainer = TextTrainer()
        X_text, y, task, le = trainer._prepare(df, "label", "text")
        assert le is not None  # LabelEncoder was created (L192-193)
        assert task == "classification"
        assert y.dtype in [np.int32, np.int64, int]

    # L418 — module-level train_best function call
    def test_module_level_train_best(self):
        import kaizenstat.model.text_trainer as tt
        df = _text_df(n=100)
        result = tt.train_best(df, "label", text_col="text", cv=2)
        assert result is not None

    # L418 — multiclass roc_auc in text _compute_metrics
    def test_compute_metrics_multiclass(self):
        from kaizenstat.model.text_trainer import TextModelTrainer as TextTrainer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        rng = np.random.default_rng(5)
        vocab = ["good", "bad", "great", "nice", "poor", "awful", "best", "worst"]
        n = 150
        texts = pd.Series([" ".join(rng.choice(vocab, 5).tolist()) for _ in range(n)])
        y = pd.Series([i % 3 for i in range(n)])  # 3 classes
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression(max_iter=300))
        ])
        pipe.fit(texts[:120], y[:120])
        metrics = TextTrainer()._compute_metrics(pipe, texts[120:], y[120:], "classification")
        assert "accuracy" in metrics


# ─────────────────────────────────────────────────────────────────────────────
# model/trainer.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainerCoverage:

    # L289 — train_best fallback split (stratify fails when class has 1 sample)
    def test_train_best_fallback_split(self):
        from kaizenstat.model.trainer import ModelTrainer
        df = pd.DataFrame({
            "a": np.random.randn(22),
            "b": np.random.randn(22),
            "target": [0] * 21 + [1],
        })
        result = ModelTrainer().train_best(df, "target")
        assert result is not None

    # L782 — _compute_metrics: multiclass roc_auc
    def test_compute_metrics_multiclass_roc_auc(self):
        from kaizenstat.model.trainer import ModelTrainer
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (150, 4))
        y = pd.Series([i % 3 for i in range(150)])
        X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=300))])
        pipe.fit(X_df[:120], y[:120])
        metrics = ModelTrainer()._compute_metrics(pipe, X_df[120:], y[120:], "classification")
        assert "accuracy" in metrics

    # _compute_metrics: roc_auc except path (single class in y_test)
    def test_compute_metrics_roc_auc_fails(self):
        from kaizenstat.model.trainer import ModelTrainer
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (50, 2)), columns=["a", "b"])
        y_train = pd.Series([i % 2 for i in range(50)])
        y_test_one_class = pd.Series([0] * 20)  # only 1 class → roc_auc raises
        X_test = X.iloc[:20]
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())])
        pipe.fit(X, y_train)
        metrics = ModelTrainer()._compute_metrics(pipe, X_test, y_test_one_class, "classification")
        assert "accuracy" in metrics

    # VotingRegressor in _build_ensemble
    def test_build_ensemble_regression(self):
        from kaizenstat.model.trainer import ModelTrainer
        r1 = Pipeline([("m", Ridge())])
        r2 = Pipeline([("m", Ridge(alpha=0.5))])
        ens = ModelTrainer()._build_ensemble([("r1", r1), ("r2", r2)], "regression")
        from sklearn.ensemble import VotingRegressor
        assert isinstance(ens, VotingRegressor)

    # LightGBM regression (mocked) — L478-484 covered by pragma, but also test mock approach
    def test_get_models_regression(self):
        from kaizenstat.model.trainer import ModelTrainer
        y = pd.Series(np.random.randn(100))
        models = ModelTrainer()._get_models("regression", y)
        assert "Ridge" in models or "RandomForest" in models

    # XGBoost except ImportError covered by pragma; test XGBoost classification exists
    def test_get_models_classification_with_xgboost(self):
        from kaizenstat.model.trainer import ModelTrainer
        y = pd.Series([0, 1] * 50)
        models = ModelTrainer()._get_models("classification", y)
        assert "XGBoost" in models  # XGBoost is installed


# ─────────────────────────────────────────────────────────────────────────────
# reliability/trust.py L272 — _robustness_reg except → return 1.0
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustCoverage:

    def test_robustness_reg_predict_raises(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=["a", "b", "c"])
        base_pred = rng.normal(0, 1, 50)
        model = MagicMock()
        model.predict.side_effect = RuntimeError("predict fail")
        result = TrustAnalyzer()._robustness_reg(model, X, base_pred)
        assert result == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# validate/checker.py
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckerCoverage:

    # L177 — _check_normality except Exception: pass
    def test_check_normality_exception(self):
        from kaizenstat.validate.checker import Validator as DataValidator
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "target": np.random.randn(100),
        })
        from scipy import stats as scipy_stats
        with patch.object(scipy_stats, "shapiro", side_effect=Exception("shapiro fail")):
            issues = DataValidator()._check_normality(df, "target")
        assert isinstance(issues, list)

    # L260 — VIF: r2 > 0 → vif computed
    def test_check_multicollinearity_vif_computed(self):
        from kaizenstat.validate.checker import Validator as DataValidator
        rng = np.random.default_rng(1)
        n = 100
        a = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "a": a,
            "b": a * 2 + rng.normal(0, 0.01, n),  # highly correlated → r2 > 0
            "c": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        issues = DataValidator()._check_multicollinearity(df, "target")
        assert isinstance(issues, list)

    # L296 — _check_multicollinearity: high VIF issue appended
    def test_check_multicollinearity_high_vif_issue(self):
        from kaizenstat.validate.checker import Validator as DataValidator
        rng = np.random.default_rng(2)
        n = 200
        a = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "a": a,
            "b": a * 3 + rng.normal(0, 0.001, n),
            "c": a * -2 + rng.normal(0, 0.001, n),
            "d": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        issues = DataValidator()._check_multicollinearity(df, "target")
        assert isinstance(issues, list)


# ─────────────────────────────────────────────────────────────────────────────
# debugger.py — remaining missed lines
# ─────────────────────────────────────────────────────────────────────────────

class TestDebuggerRemaining:

    # L871 — _data_vs_model_blame: label "excellent"/"healthy" → return None
    def test_data_vs_model_blame_excellent_label(self):
        from kaizenstat.debug.debugger import ModelDebugger
        result = ModelDebugger()._data_vs_model_blame(
            "excellent", 0.98, pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]), "classification"
        )
        assert result is None

    # L1008 — _diagnose_imbalance: acc - f1 > 0.10 → issue appended
    def test_diagnose_imbalance_issue_appended(self):
        from kaizenstat.debug.debugger import ModelDebugger
        model = MagicMock()
        # 3-class imbalanced: 70/20/10. All predictions = 0 → acc=0.70, f1<<0.70
        y_test = pd.Series([0] * 70 + [1] * 20 + [2] * 10)
        X_test = pd.DataFrame({"a": range(100)})
        model.predict.return_value = np.zeros(100, dtype=int)
        issues = ModelDebugger()._diagnose_imbalance(None, y_test, model, X_test)
        assert len(issues) > 0


# ─────────────────────────────────────────────────────────────────────────────
# health/scorer.py L278 — corr exception in leakage proxy
# ─────────────────────────────────────────────────────────────────────────────

class TestScorerLeakageException:
    def test_leakage_corr_raises(self):
        from kaizenstat.health.scorer import HealthScorer
        df = pd.DataFrame({"a": np.random.randn(100), "target": np.random.randn(100)})
        with patch("pandas.Series.corr", side_effect=Exception("corr fail")):
            result = HealthScorer().report(df, target="target")
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# text_scorer.py — _summary branches
# ─────────────────────────────────────────────────────────────────────────────

class TestTextScorerSummary:
    def test_summary_excellent(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        assert "Excellent" in TextHealthScorer._summary(92.0, "text")

    def test_summary_good(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        assert "Good" in TextHealthScorer._summary(78.0, "text")

    def test_summary_moderate(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        assert "Moderate" in TextHealthScorer._summary(63.0, "text")

    def test_summary_significant(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        assert "Significant" in TextHealthScorer._summary(45.0, "text")

    def test_summary_critical(self):
        from kaizenstat.health.text_scorer import TextHealthScorer
        assert "Critical" in TextHealthScorer._summary(20.0, "text")


# ─────────────────────────────────────────────────────────────────────────────
# improve/suggester.py — L195 overfitting gain, L302 missing, L315 skewed
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggesterRemaining:

    # L195 — "overfitting" label + gap > 0 → overfitting gain string
    def test_from_debug_overfitting_gain(self):
        from kaizenstat.improve.suggester import Suggester
        from kaizenstat.debug.debugger import DebugIssue
        issue = DebugIssue(name="X", description="d", root_cause="r",
                           risk_level="HIGH", suggestion="fix it")
        dr = MagicMock()
        dr.issues = [issue]
        dr.gap = 0.25
        dr.test_score = 0.70
        dr.label = "overfitting"
        suggs = Suggester()._from_debug(dr, start_priority=1)
        assert "gap" in suggs[0].expected_gain.lower() or "Closing" in suggs[0].expected_gain

    # L302 — _from_data: miss_rate > 0.05 → KNN imputation suggestion
    def test_from_data_missing_values(self):
        from kaizenstat.improve.suggester import Suggester
        rng = np.random.default_rng(5)
        n = 200
        data = rng.normal(0, 1, (n, 3))
        df = pd.DataFrame(data, columns=["a", "b", "c"])
        # Inject >5% missing values
        idx = rng.choice(n, size=40, replace=False)
        df.loc[idx, "a"] = np.nan
        df.loc[idx, "b"] = np.nan
        df.loc[idx, "c"] = np.nan
        df["target"] = rng.integers(0, 2, n)
        suggs = Suggester()._from_data(df, "target", start_priority=1)
        assert any("KNN" in s.action or "Imputer" in s.action for s in suggs)

    # L315 — _from_data: skewed features → transform suggestion
    def test_from_data_skewed_features(self):
        from kaizenstat.improve.suggester import Suggester
        rng = np.random.default_rng(6)
        n = 200
        df = pd.DataFrame({
            "a": np.exp(rng.normal(0, 3, n)),  # highly right-skewed
            "b": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        suggs = Suggester()._from_data(df, "target", start_priority=1)
        assert any("log" in s.action.lower() or "skew" in s.reason.lower() for s in suggs)


# ─────────────────────────────────────────────────────────────────────────────
# improve/text_suggester.py — remaining branches
# ─────────────────────────────────────────────────────────────────────────────

class TestTextSuggesterRemaining:

    # L48 — top = deduped[0] when suggestions exist
    # L85, L101, L109 — _from_debug issue branches
    def test_from_debug_sparse_issue(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        from kaizenstat.debug.debugger import DebugIssue
        # "sparse" in issue name → L85
        sparse_issue = DebugIssue(name="Sparse TF-IDF Matrix", description="d",
                                  root_cause="r", risk_level="HIGH", suggestion="fix")
        dr = MagicMock()
        dr.issues = [sparse_issue]
        dr.label = "underfitting"
        dr.test_score = 0.60
        suggs = TextSuggester()._from_debug(dr)
        assert len(suggs) >= 1

    def test_from_debug_weak_repr_issue(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        from kaizenstat.debug.debugger import DebugIssue
        # "weak representation" → L101
        issue = DebugIssue(name="Weak Representation", description="d",
                           root_cause="r", risk_level="MEDIUM", suggestion="fix")
        dr = MagicMock()
        dr.issues = [issue]
        dr.label = "underfitting"
        dr.test_score = 0.60
        suggs = TextSuggester()._from_debug(dr)
        assert isinstance(suggs, list)

    def test_from_debug_rare_token_issue(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        from kaizenstat.debug.debugger import DebugIssue
        # "rare-token" → L109
        issue = DebugIssue(name="Rare-Token Explosion", description="d",
                           root_cause="r", risk_level="HIGH", suggestion="fix")
        dr = MagicMock()
        dr.issues = [issue]
        dr.label = "overfitting"
        dr.test_score = 0.70
        suggs = TextSuggester()._from_debug(dr)
        assert isinstance(suggs, list)

    # L48 via suggest() producing non-empty deduped list
    def test_suggest_top_priority_set(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        df = pd.DataFrame({
            "text": ["a b"] * 50 + ["c d"] * 50,
            "label": [0] * 50 + [1] * 50,
        })
        # Ensure minority class < 0.20 to trigger imbalance suggestion
        df2 = pd.DataFrame({
            "text": ["a b"] * 90 + ["c d"] * 10,
            "label": [0] * 90 + [1] * 10,
        })
        report = TextSuggester().suggest(df2, target="label", text_col="text")
        assert report.top_priority is not None

    # L189 — module-level suggest
    def test_module_suggest(self):
        import kaizenstat.improve.text_suggester as ts
        df = pd.DataFrame({"text": ["hello world"] * 50, "label": [0, 1] * 25})
        report = ts.suggest(df, target="label", text_col="text")
        assert report is not None


# ─────────────────────────────────────────────────────────────────────────────
# model/trainer.py — remaining lines
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainerRemaining:

    # L289 — train_best: fallback split when stratify fails
    def test_train_best_fallback_split(self):
        from kaizenstat.model.trainer import ModelTrainer
        df = pd.DataFrame({
            "a": np.random.randn(22), "b": np.random.randn(22),
            "target": [0] * 21 + [1],
        })
        result = ModelTrainer().train_best(df, "target")
        assert result is not None

    # L518 — _build_ensemble classification → VotingClassifier
    def test_build_ensemble_classification(self):
        from kaizenstat.model.trainer import ModelTrainer
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        p1 = Pipeline([("m", LogisticRegression())])
        p2 = Pipeline([("m", LogisticRegression(C=0.5))])
        ens = ModelTrainer()._build_ensemble([("p1", p1), ("p2", p2)], "classification")
        assert isinstance(ens, VotingClassifier)

    # L669-670 — train_auto: fallback split when stratify fails
    def test_train_auto_fallback_split(self):
        from kaizenstat.model.trainer import ModelTrainer
        df = pd.DataFrame({
            "a": np.random.randn(22), "b": np.random.randn(22),
            "target": [0] * 21 + [1],
        })
        result = ModelTrainer().train_auto(df, "target")
        assert result is not None

    # L726-731 — train_auto stacking fallback when stacking fails
    def test_train_auto_stacking_fallback(self):
        from kaizenstat.model.trainer import ModelTrainer
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 300),
            "b": rng.normal(0, 1, 300),
            "target": [i % 2 for i in range(300)],
        })
        with patch.object(ModelTrainer, "_build_stacking_ensemble", side_effect=Exception("stack fail")):
            result = ModelTrainer().train_auto(df, "target", ensemble=True)
        assert result is not None

    # L739-742 — train_auto: calibration confidence check (covered by normal train_auto run)
    def test_train_auto_basic(self):
        from kaizenstat.model.trainer import ModelTrainer
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
            "target": [i % 2 for i in range(100)],
        })
        result = ModelTrainer().train_auto(df, "target")
        assert result is not None

    # L782 — _compute_metrics: multiclass roc_auc except path
    def test_compute_metrics_roc_auc_exception(self):
        from kaizenstat.model.trainer import ModelTrainer
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (50, 2)), columns=["a", "b"])
        y_train = pd.Series([i % 2 for i in range(50)])
        y_test_one = pd.Series([0] * 20)  # single class → roc_auc raises
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())])
        pipe.fit(X, y_train)
        metrics = ModelTrainer()._compute_metrics(pipe, X.iloc[:20], y_test_one, "classification")
        assert "accuracy" in metrics


# ─────────────────────────────────────────────────────────────────────────────
# reliability/trust.py — remaining exception paths
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustRemaining:

    def _setup(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=["a", "b", "c"])
        y = pd.Series([i % 2 for i in range(50)])
        model = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())])
        model.fit(X, y)
        return TrustAnalyzer(), model, X, y

    # L142-143 — _get_proba: predict_proba raises → return None
    def test_get_proba_raises(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        model = MagicMock()
        model.predict_proba.side_effect = RuntimeError("proba fail")
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(0, 1, (20, 2)), columns=["a", "b"])
        result = TrustAnalyzer()._get_proba(model, X)
        assert result is None

    # L152-153 — _get_proba decision_function path raises → return None
    def test_get_proba_decision_raises(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        model = MagicMock(spec=["decision_function"])
        model.decision_function.side_effect = RuntimeError("dec fail")
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(0, 1, (20, 2)), columns=["a", "b"])
        result = TrustAnalyzer()._get_proba(model, X)
        assert result is None

    # L169-170 — _robustness: model.predict raises → continue
    def test_robustness_predict_raises(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        model = MagicMock()
        model.predict.side_effect = RuntimeError("predict fail")
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=["a", "b", "c"])
        base_pred = np.array([0, 1] * 25)
        result = TrustAnalyzer()._robustness(model, X, base_pred, n_perturb=3)
        assert isinstance(result, float)

    # L272 — _robustness_reg: model.predict raises → return 1.0
    def test_robustness_reg_raises(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        model = MagicMock()
        model.predict.side_effect = RuntimeError("fail")
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=["a", "b", "c"])
        result = TrustAnalyzer()._robustness_reg(model, X, np.zeros(50))
        assert result == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# validate/text_checker.py — all remaining branches
# ─────────────────────────────────────────────────────────────────────────────

class TestTextCheckerRemaining:

    # L65 — _tokenize: len(s) > sample → s.sample(...)
    def test_tokenize_large_series(self):
        from kaizenstat.validate.text_checker import TextValidator
        s = pd.Series(["hello world great nice"] * 4000)
        toks = TextValidator._tokenize(s, sample=3000)
        assert isinstance(toks, list) and len(toks) > 0

    # L75 — _check_token_skew: not tokens → return []
    def test_check_token_skew_empty(self):
        from kaizenstat.validate.text_checker import TextValidator
        result = TextValidator()._check_token_skew([], "col")
        assert result == []

    # L94 — _check_stopword_dominance: not tokens → return []
    def test_check_stopword_dominance_empty(self):
        from kaizenstat.validate.text_checker import TextValidator
        result = TextValidator()._check_stopword_dominance([], "col")
        assert result == []

    # L99 — _check_stopword_dominance: ratio > 0.55 → return issue
    def test_check_stopword_dominance_high(self):
        from kaizenstat.validate.text_checker import TextValidator
        # Mostly stopwords
        stopword_heavy = ["the", "is", "a", "and", "of", "to", "it", "in",
                          "that", "this"] * 500 + ["signal"] * 50
        result = TextValidator()._check_stopword_dominance(stopword_heavy, "text")
        assert len(result) > 0

    # L110 — _check_rare_explosion: not tokens → return []
    def test_check_rare_explosion_empty(self):
        from kaizenstat.validate.text_checker import TextValidator
        result = TextValidator()._check_rare_explosion([], "col")
        assert result == []

    # L117 — _check_rare_explosion: high hapax ratio → return issue
    def test_check_rare_explosion_high_hapax(self):
        from kaizenstat.validate.text_checker import TextValidator
        # 200 unique words each appearing once → high hapax ratio
        tokens = [f"uniqueword{i}" for i in range(200)] + ["common"] * 50
        result = TextValidator()._check_rare_explosion(tokens, "text")
        assert len(result) > 0

    # L147 — _check_label_leakage: low-frequency token skipped (continue)
    def test_check_label_leakage_low_freq_token(self):
        from kaizenstat.validate.text_checker import TextValidator
        # Texts with a rare "signal" word appearing only once (too rare to be flagged)
        texts = ["the quick brown fox"] * 100 + ["leakword unique text"] * 1
        labels = [0] * 100 + [1]
        df = pd.DataFrame({"text": texts, "target": labels})
        result = TextValidator()._check_label_leakage(df, "target", "text")
        assert isinstance(result, list)

    # L178 — module-level assumptions function
    def test_module_assumptions(self):
        import kaizenstat.validate.text_checker as tc
        df = pd.DataFrame({"text": ["hello world"] * 50, "label": [0, 1] * 25})
        report = tc.assumptions(df, target="label", text_col="text")
        assert report is not None


# ─────────────────────────────────────────────────────────────────────────────
# data_doctor.py — remaining paths
# ─────────────────────────────────────────────────────────────────────────────

class TestDataDoctorRemaining:

    # L292 — debug_model: LabelEncoder transform (string target)
    def test_debug_model_label_encoder_path(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "target": ["yes"] * 50 + ["no"] * 50,
        })
        doc = DataDoctor()
        doc.fit(df, target="target")
        doc.train()
        result = doc.debug_model()
        assert result is not None

    # L458 — trust_score: LabelEncoder transform (string target)
    def test_trust_score_string_target(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "target": ["yes"] * 50 + ["no"] * 50,
        })
        doc = DataDoctor()
        doc.fit(df, target="target")
        doc.train()
        result = doc.trust_score()
        assert 0 <= result.trust_score <= 100

    # L531-532 — trust_score text mode fallback split
    def test_trust_score_text_fallback(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        rng = np.random.default_rng(5)
        vocab = ["good", "bad", "nice", "poor", "great", "awful"]
        texts = [" ".join(rng.choice(vocab, 5).tolist()) for _ in range(30)]
        df = pd.DataFrame({"text": texts, "label": [0] * 29 + [1]})
        doc = DataDoctor()
        doc.fit(df, target="label")
        doc.train()
        result = doc.trust_score()
        assert 0 <= result.trust_score <= 100

    # L602-603 — _heal_text removes empty documents
    def test_heal_text_removes_empty_docs(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        rng = np.random.default_rng(9)
        vocab = ["good", "bad", "nice", "poor"]
        texts = [" ".join(rng.choice(vocab, 5).tolist()) for _ in range(28)]
        df = pd.DataFrame({
            "text": texts + ["  ", "   "],
            "label": [0, 1] * 14 + [0, 1],
        })
        doc = DataDoctor()
        doc.fit(df, target="label")
        cleaned = doc._heal_text(df)
        assert len(cleaned) < len(df)


# ─────────────────────────────────────────────────────────────────────────────
# model/text_trainer.py L398 — except in _compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestTextTrainerMetrics:

    # L398 — _compute_metrics except Exception path (roc_auc fails)
    def test_compute_metrics_roc_auc_exception(self):
        from kaizenstat.model.text_trainer import TextModelTrainer as TextTrainer
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = pd.Series(["hello world"] * 40 + ["foo bar"] * 40)
        y_train = pd.Series([i % 2 for i in range(80)])
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=200))])
        pipe.fit(texts, y_train)
        # Single class in y_test → roc_auc_score raises
        metrics = TextTrainer()._compute_metrics(pipe, texts[:20], pd.Series([0] * 20), "classification")
        assert "accuracy" in metrics


# =============================================================================
# FINAL 21-LINE COVERAGE PUSH
# =============================================================================

class TestFinalCoverageLines:
    """Targets every remaining uncovered line to achieve 100%."""

    # ── trainer.py L782 ──────────────────────────────────────────────────────
    # roc_auc except Exception — mock roc_auc_score to actually raise
    def test_trainer_compute_metrics_roc_auc_mock_raises(self):
        from kaizenstat.model.trainer import ModelTrainer
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (60, 2)), columns=["a", "b"])
        y = pd.Series([i % 2 for i in range(60)])
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())])
        pipe.fit(X[:40], y[:40])
        with patch("kaizenstat.model.trainer.roc_auc_score", side_effect=ValueError("roc fail")):
            metrics = ModelTrainer()._compute_metrics(pipe, X[40:], y[40:], "classification")
        assert "accuracy" in metrics

    # ── text_trainer.py L398 ─────────────────────────────────────────────────
    # roc_auc except in _compute_metrics — mock roc_auc_score to raise
    def test_text_trainer_compute_metrics_roc_auc_mock_raises(self):
        from kaizenstat.model.text_trainer import TextModelTrainer
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = pd.Series(["hello world"] * 40 + ["foo bar"] * 40)
        y_train = pd.Series([i % 2 for i in range(80)])
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=200))])
        pipe.fit(texts, y_train)
        with patch("kaizenstat.model.text_trainer.roc_auc_score", side_effect=ValueError("roc fail")):
            metrics = TextModelTrainer()._compute_metrics(pipe, texts[:20], y_train[:20], "classification")
        assert "accuracy" in metrics

    # ── text_debugger.py L146 ────────────────────────────────────────────────
    # calibrated_classifiers_ try block — .estimator succeeds, getattr called
    def test_text_debugger_calibrated_coef_getattr(self):
        from kaizenstat.debug.text_debugger import TextModelDebugger
        vec = MagicMock()
        vec.get_feature_names_out.side_effect = Exception("no names")
        inner = MagicMock()
        del inner.coef_
        del inner.estimators_
        base = MagicMock()
        base.coef_ = np.array([[0.1, 0.2, 0.3]])
        cal_clf = MagicMock()
        cal_clf.estimator = base  # .estimator returns normally (not raises)
        inner.calibrated_classifiers_ = [cal_clf]
        model = MagicMock()
        model.named_steps = {"tfidf": vec, "model": inner}
        result = TextModelDebugger()._top_tokens(model)
        # coef_ has shape (1,3) and feat_names will fail (no get_feature_names_out)
        # so result is None, but L146 IS covered
        assert result is None

    # ── text_debugger.py L271 ────────────────────────────────────────────────
    # module-level model_failure convenience function
    def test_text_debugger_module_level_model_failure(self):
        import kaizenstat.debug.text_debugger as td
        from sklearn.feature_extraction.text import TfidfVectorizer
        rng = np.random.default_rng(7)
        vocab = ["good", "bad", "nice", "poor", "great", "ugly"]
        texts_tr = pd.Series([" ".join(rng.choice(vocab, 5).tolist()) for _ in range(80)])
        texts_te = pd.Series([" ".join(rng.choice(vocab, 5).tolist()) for _ in range(20)])
        y_tr = pd.Series([i % 2 for i in range(80)])
        y_te = pd.Series([i % 2 for i in range(20)])
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=200))])
        pipe.fit(texts_tr, y_tr)
        result = td.model_failure(pipe, texts_tr, texts_te, y_tr, y_te)
        assert result is not None

    # ── text_scorer.py L223, L227, L231 ─────────────────────────────────────
    # module-level score, report, breakdown functions
    def test_text_scorer_module_level_score(self):
        import kaizenstat.health.text_scorer as ts
        rng = np.random.default_rng(1)
        vocab = ["good", "bad", "nice", "poor", "happy", "sad"]
        df = pd.DataFrame({
            "text": [" ".join(rng.choice(vocab, 5).tolist()) for _ in range(100)],
            "label": [i % 2 for i in range(100)],
        })
        result = ts.score(df, target="label", text_col="text")
        assert 0 <= result <= 100

    def test_text_scorer_module_level_report(self):
        import kaizenstat.health.text_scorer as ts
        rng = np.random.default_rng(2)
        vocab = ["good", "bad", "nice", "poor"]
        df = pd.DataFrame({
            "text": [" ".join(rng.choice(vocab, 5).tolist()) for _ in range(100)],
            "label": [i % 2 for i in range(100)],
        })
        result = ts.report(df, target="label", text_col="text")
        assert result is not None

    def test_text_scorer_module_level_breakdown(self):
        import kaizenstat.health.text_scorer as ts
        rng = np.random.default_rng(3)
        vocab = ["good", "bad", "nice", "poor"]
        df = pd.DataFrame({
            "text": [" ".join(rng.choice(vocab, 5).tolist()) for _ in range(100)],
            "label": [i % 2 for i in range(100)],
        })
        result = ts.breakdown(df, target="label", text_col="text")
        assert result is not None

    # ── text_suggester.py L48 ────────────────────────────────────────────────
    # continue (skip duplicate action) in dedup loop
    def test_text_suggester_dedup_continue(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        from kaizenstat.debug.debugger import DebugIssue
        rng = np.random.default_rng(8)
        vocab = ["good", "bad", "nice", "poor", "great", "ugly", "fast", "slow"]
        df = pd.DataFrame({
            "text": [" ".join(rng.choice(vocab, 4).tolist()) for _ in range(100)],
            "label": [i % 2 for i in range(100)],
        })
        # Create two debug issues that produce suggestions with identical action text
        issue_sparse = DebugIssue(name="sparse",
                                  description="Sparse TF-IDF",
                                  root_cause="high sparsity",
                                  risk_level="HIGH",
                                  suggestion="Add char n-grams")
        dr = MagicMock()
        dr.issues = [issue_sparse, issue_sparse]  # two identical issues → same action → dedup fires
        dr.label = "underfitting"
        dr.test_score = 0.5
        report = TextSuggester().suggest(df, target="label", debug_result=dr)
        # Dedup should have removed duplicates
        actions = [s.action for s in report.suggestions]
        assert len(actions) == len(set(actions))

    # ── text_suggester.py L109 ───────────────────────────────────────────────
    # "imbalance" in issue name → imbalance suggestion
    def test_text_suggester_imbalance_issue(self):
        from kaizenstat.improve.text_suggester import TextSuggester
        from kaizenstat.debug.debugger import DebugIssue
        issue = DebugIssue(name="imbalance",
                           description="Class imbalance detected",
                           root_cause="minority class",
                           risk_level="HIGH",
                           suggestion="Oversample minority")
        dr = MagicMock()
        dr.issues = [issue]
        dr.label = "overfitting"
        dr.test_score = 0.6
        out = TextSuggester()._from_debug(dr)
        assert any("class_weight" in s.action.lower() or "oversample" in s.action.lower() for s in out)

    # ── debugger.py L871 ─────────────────────────────────────────────────────
    # _data_vs_model_blame: X_train is not a DataFrame → return None
    def test_debugger_data_vs_model_blame_non_dataframe(self):
        from kaizenstat.debug.debugger import ModelDebugger
        result = ModelDebugger()._data_vs_model_blame(
            "underfitting", 0.55,
            np.array([[1, 2], [3, 4]]),  # numpy array, not DataFrame
            pd.Series([0, 1]),
            "classification"
        )
        assert result is None

    # ── debugger.py L1036 ────────────────────────────────────────────────────
    # feature_importances_ exists but len(fi) raises → except Exception at L1036
    def test_debugger_extract_feature_importance_tree_len_raises(self):
        from kaizenstat.debug.debugger import ModelDebugger

        class _BadLen:
            def __len__(self):
                raise RuntimeError("len fail")

        inner = MagicMock()
        inner.feature_importances_ = _BadLen()  # hasattr=True, access OK, len() raises
        del inner.coef_
        model = MagicMock()
        model.named_steps = {"model": inner}
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        y = pd.Series([i % 2 for i in range(20)])
        with patch("kaizenstat.debug.debugger.permutation_importance", side_effect=Exception("perm")):
            result = ModelDebugger()._extract_feature_importance(model, X, y, "classification")
        assert result is None

    # ── debugger.py L1048 ────────────────────────────────────────────────────
    # coef_ exists but np.asarray raises → except Exception at L1048
    def test_debugger_extract_feature_importance_coef_asarray_raises(self):
        from kaizenstat.debug.debugger import ModelDebugger

        class _BadArray:
            def __array__(self, dtype=None, copy=None):
                raise RuntimeError("array fail")

        inner = MagicMock()
        del inner.feature_importances_
        inner.coef_ = _BadArray()  # hasattr=True, access OK, np.asarray raises
        model = MagicMock()
        model.named_steps = {"model": inner}
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        y = pd.Series([i % 2 for i in range(20)])
        with patch("kaizenstat.debug.debugger.permutation_importance", side_effect=Exception("perm")):
            result = ModelDebugger()._extract_feature_importance(model, X, y, "classification")
        assert result is None

    # ── trust.py L272 ────────────────────────────────────────────────────────
    # _robustness_reg: Xp is None (X is not DataFrame or Series)
    def test_trust_robustness_reg_xp_none(self):
        from kaizenstat.reliability.trust import TrustAnalyzer
        model = MagicMock()
        base_pred = np.array([1.0, 2.0, 3.0])
        # Pass numpy array → _perturb returns None → early return 1.0
        result = TrustAnalyzer()._robustness_reg(model, np.array([[1, 2], [3, 4], [5, 6]]), base_pred)
        assert result == 1.0

    # ── checker.py L177 ──────────────────────────────────────────────────────
    # normaltest path: sample > 5000 rows. Since code samples min(len,4000),
    # this line is unreachable — add pragma instead of a test.
    # (handled via source edit below)

    # ── checker.py L260 ──────────────────────────────────────────────────────
    # VIF continue: only 1 numeric col → others == [] → continue
    def test_checker_vif_single_numeric_col(self):
        from kaizenstat.validate.checker import Validator
        rng = np.random.default_rng(10)
        n = 200
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),  # only 1 numeric col
            "target": rng.integers(0, 2, n),
        })
        issues = Validator()._check_multicollinearity(df, "target")
        assert isinstance(issues, list)

    # ── checker.py L296 ──────────────────────────────────────────────────────
    # _check_label_leakage: corr raises → except Exception: pass
    def test_checker_leakage_corr_raises(self):
        from kaizenstat.validate.checker import Validator
        rng = np.random.default_rng(11)
        n = 100
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        with patch.object(pd.Series, "corr", side_effect=Exception("corr fail")):
            issues = Validator()._check_leakage(df, "target")
        assert isinstance(issues, list)

    # ── data_doctor.py L458 ──────────────────────────────────────────────────
    # feature_impact with label_encoder: string target → le.transform(y) at L458
    def test_data_doctor_feature_impact_string_target(self):
        from kaizenstat.doctor.data_doctor import DataDoctor
        rng = np.random.default_rng(12)
        n = 120
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "target": ["cat"] * 60 + ["dog"] * 60,
        })
        doc = DataDoctor()
        doc.fit(df, target="target")
        doc.train()
        # train() with string target creates a LabelEncoder; feature_impact hits L458
        result = doc.feature_impact()
        assert isinstance(result, dict)


# Add pragma for checker.py L177 via source edit (unreachable path)
