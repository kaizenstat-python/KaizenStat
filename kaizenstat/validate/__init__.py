"""
validate — Statistical assumption and leakage validation module.

Usage::

    from kaizenstat import validate

    validate.assumptions(df, target="y")
    validate.skewness(df)
    validate.multicollinearity(df)
    validate.leakage(df, target="y")
    validate.distribution_check(df)
"""
from .checker import (
    Validator,
    ValidationReport,
    ValidationIssue,
    assumptions,
    skewness,
    multicollinearity,
    leakage,
    distribution_check,
    detect_drift,
)
from . import text_checker
from .text_checker import TextValidator

__all__ = [
    "Validator",
    "ValidationReport",
    "ValidationIssue",
    "assumptions",
    "skewness",
    "multicollinearity",
    "leakage",
    "distribution_check",
    "detect_drift",
    "text_checker",
    "TextValidator",
]
