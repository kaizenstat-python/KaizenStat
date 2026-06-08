"""Optional AI advisor — uses Anthropic Claude when API key is provided."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class AIAdvisor:
    """
    Provides AI-generated advice using Anthropic Claude.

    Usage:
        advisor = AIAdvisor(api_key="sk-ant-...")
        advisor.advise(health_result=hr, debug_result=dr)
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to initialise Anthropic client: {exc}") from exc
        return self._client

    def advise(
        self,
        health_result: Optional[Any] = None,
        debug_result: Optional[Any] = None,
        validation_result: Optional[Any] = None,
        question: Optional[str] = None,
    ) -> str:
        """Ask Claude for tailored advice based on pipeline results."""
        if not self.api_key:
            console.print(
                "[yellow]⚠ No Anthropic API key provided. "
                "Set ANTHROPIC_API_KEY or pass api_key= to AIAdvisor().[/yellow]"
            )
            return ""

        context = self._build_context(health_result, debug_result, validation_result)
        prompt = self._build_prompt(context, question)

        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response = msg.content[0].text
            console.print(Panel(
                Markdown(response),
                title="[bold cyan]AI Advisor (Claude)[/bold cyan]",
                border_style="cyan",
            ))
            return response
        except Exception as exc:
            console.print(f"[red]AI Advisor error: {exc}[/red]")
            return ""

    def ask(self, question: str) -> str:
        """Ask a free-form question about ML/data health."""
        return self.advise(question=question)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(health_result, debug_result, validation_result) -> Dict:
        ctx: Dict[str, Any] = {}

        if health_result is not None:
            ctx["health"] = {
                "score": health_result.score,
                "grade": health_result.grade,
                "risk": health_result.risk_level,
                "penalties": [
                    {"name": p.name, "penalty": p.penalty, "risk": p.risk_level}
                    for p in health_result.penalties
                ],
            }

        if debug_result is not None:
            ctx["debug"] = {
                "train_score": debug_result.train_score,
                "test_score": debug_result.test_score,
                "gap": debug_result.gap,
                "diagnosis": debug_result.diagnosis,
                "root_cause": debug_result.root_cause,
                "issues": [i.name for i in debug_result.issues],
            }

        if validation_result is not None:
            ctx["validation"] = {
                "passed": validation_result.passed,
                "issues": [
                    {"check": i.check, "risk": i.risk_level, "issue": i.issue}
                    for i in validation_result.issues
                ],
            }

        return ctx

    @staticmethod
    def _build_prompt(context: Dict, question: Optional[str]) -> str:
        ctx_str = json.dumps(context, indent=2) if context else "(no pipeline results available)"
        base = (
            "You are KaizenStat's AI advisor — an expert in data quality and ML model debugging. "
            "Analyse the following pipeline results and provide concise, actionable recommendations "
            "in plain English. Focus on the highest-impact issues first.\n\n"
            f"Pipeline Context:\n```json\n{ctx_str}\n```\n\n"
        )
        if question:
            base += f"User question: {question}"
        else:
            base += (
                "Provide: (1) the most critical issue, (2) root cause, "
                "(3) exact next steps to improve model performance."
            )
        return base


# ------------------------------------------------------------------ #
# Module-level convenience API
# ------------------------------------------------------------------ #
_advisor: Optional[AIAdvisor] = None


def init(api_key: Optional[str] = None, model: Optional[str] = None) -> AIAdvisor:
    """Initialise the global AI advisor."""
    global _advisor
    _advisor = AIAdvisor(api_key=api_key, model=model)
    console.print("[green]✓ AI Advisor initialised.[/green]")
    return _advisor


def advise(
    health_result=None,
    debug_result=None,
    validation_result=None,
    question: Optional[str] = None,
) -> str:
    if _advisor is None:
        console.print("[yellow]Call intelligence.init(api_key=...) first.[/yellow]")
        return ""
    return _advisor.advise(
        health_result=health_result,
        debug_result=debug_result,
        validation_result=validation_result,
        question=question,
    )


def ask(question: str) -> str:
    if _advisor is None:
        console.print("[yellow]Call intelligence.init(api_key=...) first.[/yellow]")
        return ""
    return _advisor.ask(question)
