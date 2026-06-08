"""Tests for intelligence/ai_advisor.py — AIAdvisor, module-level API."""
import pytest

from kaizenstat.intelligence import ai_advisor as ai_mod
from kaizenstat.intelligence.ai_advisor import AIAdvisor


# ── Minimal stub objects ──────────────────────────────────────────────────────

class _Penalty:
    name = "Missing Values"
    penalty = -5.0
    risk_level = "MEDIUM"


class _HealthResult:
    score = 85
    grade = "B"
    risk_level = "LOW"
    penalties = [_Penalty()]


class _DebugIssue:
    name = "Overfitting"


class _DebugResult:
    train_score = 0.95
    test_score = 0.70
    gap = 0.25
    diagnosis = "overfitting"
    root_cause = "Too complex"
    issues = [_DebugIssue()]


class _ValIssue:
    check = "Normality"
    risk_level = "MEDIUM"
    issue = "Non-normal"


class _ValidationResult:
    passed = True
    issues = [_ValIssue()]


# ── AIAdvisor without API key ─────────────────────────────────────────────────

class TestAIAdvisorNoKey:
    def test_init_no_key(self):
        advisor = AIAdvisor(api_key=None)
        assert advisor.api_key is None or advisor.api_key == ""

    def test_advise_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        advisor = AIAdvisor(api_key=None)
        advisor.api_key = None
        result = advisor.advise()
        assert result == ""

    def test_ask_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        advisor = AIAdvisor(api_key=None)
        advisor.api_key = None
        result = advisor.ask("What should I do?")
        assert result == ""

    def test_init_with_model_override(self):
        advisor = AIAdvisor(api_key=None, model="claude-haiku-4-5-20251001")
        assert advisor.model == "claude-haiku-4-5-20251001"

    def test_default_model_set(self):
        advisor = AIAdvisor(api_key=None)
        assert advisor.model == AIAdvisor.DEFAULT_MODEL


# ── _build_context ────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_empty_context(self):
        ctx = AIAdvisor._build_context(None, None, None)
        assert ctx == {}

    def test_health_context(self):
        ctx = AIAdvisor._build_context(_HealthResult(), None, None)
        assert "health" in ctx
        assert ctx["health"]["score"] == 85
        assert ctx["health"]["grade"] == "B"
        assert len(ctx["health"]["penalties"]) == 1

    def test_debug_context(self):
        ctx = AIAdvisor._build_context(None, _DebugResult(), None)
        assert "debug" in ctx
        assert ctx["debug"]["train_score"] == 0.95
        assert "Overfitting" in ctx["debug"]["issues"]

    def test_validation_context(self):
        ctx = AIAdvisor._build_context(None, None, _ValidationResult())
        assert "validation" in ctx
        assert ctx["validation"]["passed"] is True
        assert len(ctx["validation"]["issues"]) == 1

    def test_full_context(self):
        ctx = AIAdvisor._build_context(_HealthResult(), _DebugResult(), _ValidationResult())
        assert "health" in ctx
        assert "debug" in ctx
        assert "validation" in ctx


# ── _build_prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_prompt_with_question(self):
        prompt = AIAdvisor._build_prompt({}, "How can I improve accuracy?")
        assert "How can I improve accuracy?" in prompt
        assert "KaizenStat" in prompt

    def test_prompt_no_question(self):
        prompt = AIAdvisor._build_prompt({"health": {"score": 80}}, None)
        assert "most critical issue" in prompt or "next steps" in prompt

    def test_prompt_empty_context(self):
        prompt = AIAdvisor._build_prompt({}, None)
        assert "no pipeline results" in prompt or "KaizenStat" in prompt

    def test_prompt_contains_context_json(self):
        ctx = {"health": {"score": 75}}
        prompt = AIAdvisor._build_prompt(ctx, None)
        assert "75" in prompt


# ── _get_client raises ImportError ───────────────────────────────────────────

class TestGetClient:
    def test_get_client_no_anthropic_raises(self, monkeypatch):
        """When anthropic package is missing, _get_client should raise ImportError."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        advisor = AIAdvisor(api_key="fake-key")
        advisor._client = None
        with pytest.raises(ImportError, match="anthropic"):
            advisor._get_client()


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_advise_without_init_returns_empty(self):
        ai_mod._advisor = None
        result = ai_mod.advise()
        assert result == ""

    def test_ask_without_init_returns_empty(self):
        ai_mod._advisor = None
        result = ai_mod.ask("Question?")
        assert result == ""

    def test_init_creates_advisor(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        advisor = ai_mod.init(api_key=None)
        assert isinstance(advisor, AIAdvisor)
        assert ai_mod._advisor is advisor

    def test_advise_after_init_no_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ai_mod.init(api_key=None)
        ai_mod._advisor.api_key = None
        result = ai_mod.advise()
        assert result == ""

    def test_ask_after_init_no_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ai_mod.init(api_key=None)
        ai_mod._advisor.api_key = None
        result = ai_mod.ask("What is overfitting?")
        assert result == ""
