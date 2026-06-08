"""
fix — Safe data correction engine.

Usage::

    from kaizenstat import fix

    plan = fix.plan(df, target="y", safe=True)   # preview
    df_clean = plan.apply(df)                     # apply

    # Or in one step:
    df_clean = fix.apply(df, target="y")
"""
from .engine import (
    FixEngine,
    FixPlan,
    FixAction,
    plan,
    apply,
    missing,
    outlier_handling,
    encoding,
    imbalance,
)

__all__ = [
    "FixEngine",
    "FixPlan",
    "FixAction",
    "plan",
    "apply",
    "missing",
    "outlier_handling",
    "encoding",
    "imbalance",
]
