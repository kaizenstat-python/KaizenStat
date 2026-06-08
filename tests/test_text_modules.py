"""Tests for all text (NLP) modules: text_scorer, text_trainer, text_debugger,
text_suggester, text_checker, and DataDoctor in text mode."""
import pandas as pd
import pytest

from kaizenstat.doctor.data_doctor import DataDoctor


# ── Text dataset fixture ───────────────────────────────────────────────────────

@pytest.fixture
def text_df_large():
    """100-row text classification dataset with enough unique sentences."""
    reviews = [
        "This product is absolutely amazing and works exactly as described",
        "Terrible quality waste of money I want a full refund immediately",
        "Great service staff were friendly helpful and very professional",
        "Awful experience never returning to this store again very disappointed",
        "Perfect item fast shipping exactly what I needed great value",
        "Mediocre performance nothing special but gets the basic job done",
        "Outstanding quality best purchase I have made this entire year",
        "Disappointing results did not meet any of my high expectations",
        "Excellent build quality very durable worth every single penny",
        "Poor customer support took many weeks to resolve my simple issue",
    ]
    data = []
    for i in range(100):
        # vary each sentence slightly to avoid all-same text
        text = reviews[i % 10] + f" item number {i}"
        data.append({"review": text, "sentiment": i % 2})
    return pd.DataFrame(data)


# ── TextHealthScorer ──────────────────────────────────────────────────────────

from kaizenstat.health.text_scorer import TextHealthScorer


class TestTextHealthScorer:
    def test_score_returns_float(self, text_df_large):
        score = TextHealthScorer().score(text_df_large, target="sentiment", text_col="review")
        assert 0 <= score <= 100

    def test_report_returns_health_result(self, text_df_large):
        result = TextHealthScorer().report(text_df_large, target="sentiment", text_col="review")
        assert 0 <= result.score <= 100

    def test_breakdown_no_text_col_auto_detects(self, text_df_large):
        result = TextHealthScorer().breakdown(text_df_large, target="sentiment")
        assert result is not None

    def test_breakdown_with_duplicates(self):
        """Heavy duplicate text triggers duplicate penalty."""
        df = pd.DataFrame({
            "review": ["same text"] * 60 + [f"unique text {i}" for i in range(40)],
            "sentiment": [i % 2 for i in range(100)],
        })
        result = TextHealthScorer().breakdown(df, target="sentiment", text_col="review")
        assert result.score <= 100

    def test_breakdown_with_noisy_text(self):
        """HTML tags and URLs trigger noise penalty."""
        texts = [f"<b>Click here</b> https://example.com/page{i} buy now !!! @@@" for i in range(100)]
        df = pd.DataFrame({"review": texts, "sentiment": [i % 2 for i in range(100)]})
        result = TextHealthScorer().breakdown(df, target="sentiment", text_col="review")
        assert result.score <= 100

    def test_breakdown_no_target(self, text_df_large):
        result = TextHealthScorer().breakdown(text_df_large, text_col="review")
        assert result is not None

    def test_breakdown_with_short_texts(self):
        """Very short texts trigger short/empty penalty."""
        df = pd.DataFrame({
            "review": ["ok", "bad", "good", "meh", "fine"] * 20,
            "sentiment": [i % 2 for i in range(100)],
        })
        result = TextHealthScorer().breakdown(df, target="sentiment", text_col="review")
        assert result is not None

    def test_breakdown_no_text_col_raises(self):
        """When no text col found, raises ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
        with pytest.raises(ValueError, match="No dominant text column"):
            TextHealthScorer().breakdown(df, target="target")

    def test_breakdown_imbalanced_text_classification(self):
        """Heavy class imbalance triggers imbalance penalty."""
        reviews = [f"review text sample number {i} with many words" for i in range(100)]
        df = pd.DataFrame({
            "review": reviews,
            "sentiment": [0] * 95 + [1] * 5,
        })
        result = TextHealthScorer().breakdown(df, target="sentiment", text_col="review")
        assert result.score <= 100


# ── TextValidator ─────────────────────────────────────────────────────────────

from kaizenstat.validate.text_checker import TextValidator


class TestTextValidator:
    def test_assumptions_returns_report(self, text_df_large):
        result = TextValidator().assumptions(text_df_large, target="sentiment", text_col="review")
        assert result.checks_run > 0

    def test_assumptions_auto_detect_text_col(self, text_df_large):
        result = TextValidator().assumptions(text_df_large, target="sentiment")
        assert result is not None

    def test_assumptions_no_text_found_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
        with pytest.raises(ValueError, match="No dominant text column"):
            TextValidator().assumptions(df, target="target")

    def test_display_runs(self, text_df_large):
        result = TextValidator().assumptions(text_df_large, target="sentiment", text_col="review")
        result.display()

    def test_assumptions_with_leakage_text(self):
        """Text column containing class labels triggers leakage warning."""
        df = pd.DataFrame({
            "review": ["positive sentiment five star"] * 50 + ["negative sentiment one star"] * 50,
            "sentiment": [1] * 50 + [0] * 50,
        })
        result = TextValidator().assumptions(df, target="sentiment", text_col="review")
        assert result is not None

    def test_token_skew_check_with_imbalance(self):
        """Documents with very different lengths trigger token-length skew check."""
        short_docs = ["bad"] * 50
        long_docs = [f"This is a very long and detailed review with lots of words number {i}" for i in range(50)]
        df = pd.DataFrame({
            "review": short_docs + long_docs,
            "sentiment": [i % 2 for i in range(100)],
        })
        result = TextValidator().assumptions(df, target="sentiment", text_col="review")
        assert result is not None


# ── TextModelTrainer ──────────────────────────────────────────────────────────

from kaizenstat.model.text_trainer import TextModelTrainer


class TestTextModelTrainer:
    def test_train_best_returns_result(self, text_df_large):
        result = TextModelTrainer().train_best(
            text_df_large, target="sentiment", text_col="review", cv=3
        )
        assert result.task == "classification"
        assert result.test_score >= 0

    def test_train_best_auto_detect_text_col(self, text_df_large):
        result = TextModelTrainer().train_best(
            text_df_large, target="sentiment", cv=3
        )
        assert result is not None

    def test_train_best_with_tune(self, text_df_large):
        result = TextModelTrainer().train_best(
            text_df_large, target="sentiment", text_col="review",
            cv=3, tune=True, n_iter=3
        )
        assert result is not None

    def test_train_best_pipeline_has_predict(self, text_df_large):
        result = TextModelTrainer().train_best(
            text_df_large, target="sentiment", text_col="review", cv=3
        )
        assert hasattr(result.pipeline, "predict")

    def test_train_best_cv_score_positive(self, text_df_large):
        result = TextModelTrainer().train_best(
            text_df_large, target="sentiment", text_col="review", cv=3
        )
        assert result.cv_score >= 0


# ── TextModelDebugger ─────────────────────────────────────────────────────────

from kaizenstat.debug.text_debugger import TextModelDebugger


class TestTextModelDebugger:
    @pytest.fixture
    def trained_text_model(self, text_df_large):
        result = TextModelTrainer().train_best(
            text_df_large, target="sentiment", text_col="review", cv=3
        )
        return result, text_df_large

    def test_model_failure_returns_debug_result(self, trained_text_model):
        result, df = trained_text_model
        X = df["review"].fillna("").astype(str)
        y = df["sentiment"]
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        dr = TextModelDebugger().model_failure(result.pipeline, X_tr, X_te, y_tr, y_te)
        assert dr is not None
        assert hasattr(dr, "label")
        assert hasattr(dr, "test_score")


# ── TextSuggester ─────────────────────────────────────────────────────────────

from kaizenstat.improve.text_suggester import TextSuggester


class TestTextSuggester:
    def test_suggest_returns_report(self, text_df_large):
        result = TextSuggester().suggest(
            text_df_large, target="sentiment", text_col="review"
        )
        assert result is not None
        assert hasattr(result, "suggestions")

    def test_suggest_with_health_result(self, text_df_large):
        health = TextHealthScorer().breakdown(text_df_large, target="sentiment", text_col="review")
        result = TextSuggester().suggest(
            text_df_large, target="sentiment", text_col="review",
            health_result=health
        )
        assert result is not None

    def test_suggest_display_runs(self, text_df_large):
        result = TextSuggester().suggest(
            text_df_large, target="sentiment", text_col="review"
        )
        result.display()


# ── DataDoctor in text mode ───────────────────────────────────────────────────

class TestDataDoctorTextMode:
    def test_fit_detects_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        assert doc.mode() == "text"

    def test_health_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        result = doc.health()
        assert 0 <= result.score <= 100

    def test_validate_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        result = doc.validate()
        assert result is not None

    def test_train_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        result = doc.train(cv=3)
        assert result.task == "classification"

    def test_debug_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        doc.train(cv=3)
        result = doc.debug_model()
        assert result is not None

    def test_improve_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        doc.health()
        result = doc.improve()
        assert result is not None

    def test_trust_score_text_mode(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        doc.train(cv=3)
        doc.debug_model()
        result = doc.trust_score()
        assert 0 <= result.trust_score <= 100

    def test_auto_improve_text(self, text_df_large):
        doc = DataDoctor()
        doc.fit(text_df_large, target="sentiment")
        from kaizenstat.doctor.data_doctor import ComparisonResult
        result = doc.auto_improve_text(tune=False)
        assert isinstance(result, ComparisonResult)

    def test_auto_improve_text_requires_text_mode(self, small_df):
        doc = DataDoctor()
        doc.fit(small_df, target="churn")
        with pytest.raises(RuntimeError, match="text"):
            doc.auto_improve_text()
