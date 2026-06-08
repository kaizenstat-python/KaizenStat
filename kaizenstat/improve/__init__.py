"""
improve — Rule-based improvement suggestion engine.

Usage::

    from kaizenstat import improve

    improve.suggest(df, target="y", health_result=hr, debug_result=dr)
    improve.feature_engineering(df, target="y")
    improve.prioritize(suggestions)
"""
from .suggester import (
    Suggester,
    ImprovementReport,
    Suggestion,
    suggest,
    prioritize,
    feature_engineering,
)
from . import text_suggester
from .text_suggester import TextSuggester

__all__ = [
    "Suggester",
    "ImprovementReport",
    "Suggestion",
    "suggest",
    "prioritize",
    "feature_engineering",
    "text_suggester",
    "TextSuggester",
]
