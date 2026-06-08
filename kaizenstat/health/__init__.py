"""
health — Data Health Scoring module.

Usage::

    from kaizenstat import health

    health.score(df)               # → float (0–100)
    health.report(df, target="y")  # → HealthResult (prints to terminal)
    health.breakdown(df)           # → HealthResult (silent)
"""
from .scorer import HealthScorer, HealthResult, HealthPenalty, score, report, breakdown
from . import text_scorer
from .text_scorer import TextHealthScorer

__all__ = [
    "HealthScorer", "HealthResult", "HealthPenalty",
    "score", "report", "breakdown",
    "text_scorer", "TextHealthScorer",
]
