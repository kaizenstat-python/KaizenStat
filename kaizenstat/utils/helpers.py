"""Shared utility functions used across all KaizenStat modules."""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd


def detect_task_type(y: pd.Series) -> str:
    """Return 'classification' or 'regression' from target series."""
    dtype = y.dtype
    # String / category / unknown extension types → classification
    if dtype == "object" or dtype.name in ("category", "string"):
        return "classification"
    if not hasattr(dtype, 'kind'):
        return "classification"
    if dtype.kind == 'f':
        return "regression"
    if dtype.kind in ('i', 'u') and y.nunique() <= 20:
        return "classification"
    return "regression"


def get_numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols


def get_categorical_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def risk_to_color(risk: str) -> str:
    mapping = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "bold red"}
    return mapping.get(risk.upper(), "white")


def validate_dataframe(df: pd.DataFrame, target: Optional[str] = None) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if target and target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if len(df) < 5:
        warnings.warn("DataFrame has fewer than 5 rows — results may be unreliable.", stacklevel=3)


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that look like row IDs or unique identifiers."""
    id_patterns = {"id", "uuid", "index", "key", "serial", "row_num", "rowid"}
    id_cols = []
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in id_patterns or col_lower.endswith("_id") or col_lower.startswith("id_"):
            id_cols.append(col)
            continue
        _d = df[col].dtype
        _is_str_or_int = _d == "object" or (hasattr(_d, 'kind') and _d.kind in ('i', 'u'))
        if _is_str_or_int:
            if df[col].nunique() > len(df) * 0.95 and len(df) > 20:
                id_cols.append(col)
    return id_cols


# ---------------------------------------------------------------------------- #
# Text / NLP detection                                                          #
# ---------------------------------------------------------------------------- #

def _avg_word_count(series: pd.Series, sample: int = 500) -> float:
    """Average number of whitespace-delimited tokens in a string column."""
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    if len(s) > sample:
        s = s.sample(sample, random_state=42)
    return float(s.str.split().str.len().mean())


def detect_text_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
    min_avg_words: float = 3.0,
    min_avg_chars: float = 20.0,
) -> List[str]:
    """
    Detect free-text columns (sentences/documents) vs short categorical strings.

    A column qualifies as text when it is object dtype and has either a high
    average word count or a high average character length, and is not behaving
    like a low-cardinality category.
    """
    exclude = set(exclude or [])
    text_cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype != "object":
            continue
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        avg_words = _avg_word_count(s)
        avg_chars = float(s.str.len().mean())
        # Reject ID-like / near-unique short codes
        looks_categorical = s.nunique() <= max(2, min(20, len(s) * 0.05))
        if (avg_words >= min_avg_words or avg_chars >= min_avg_chars) and not looks_categorical:
            text_cols.append(col)
    return text_cols


def dominant_text_column(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> Optional[str]:
    """Return the text column with the highest average token count, or None."""
    candidates = detect_text_columns(df, exclude=exclude)
    if not candidates:
        return None
    return max(candidates, key=lambda c: _avg_word_count(df[c]))


def is_text_dataset(df: pd.DataFrame, target: Optional[str] = None) -> bool:
    """True when the dataset is dominated by a free-text feature column."""
    return dominant_text_column(df, exclude=[target] if target else None) is not None
