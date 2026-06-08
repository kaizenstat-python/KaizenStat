"""Pytest fixtures and shared test utilities."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_df():
    """Minimal dataset (50 rows) for fast tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'age': rng.integers(20, 65, 50),
        'income': rng.normal(50000, 15000, 50),
        'employed': rng.integers(0, 2, 50),
        'churn': rng.integers(0, 2, 50),
    })


@pytest.fixture
def small_df():
    """Small dataset (500 rows) — truly clean: no leakage, no imbalance."""
    rng = np.random.default_rng(42)
    n = 500
    # Round income to 2 decimal places to avoid float all-unique triggering old leakage heuristic,
    # but the real fix is in the validator (float columns are never ID-like).
    income = np.round(rng.normal(55000, 20000, n), 2)
    return pd.DataFrame({
        'age': rng.integers(18, 80, n),
        'income': income,
        'tenure': rng.integers(0, 10, n),
        'employed': rng.choice(['yes', 'no'], n),
        'city': rng.choice(['NY', 'LA', 'Chicago'], n),
        'churn': rng.integers(0, 2, n),
    })


@pytest.fixture
def imbalanced_df():
    """Class-imbalanced dataset (95% / 5%) — minority clearly below 10% threshold."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        'feature1': rng.normal(0, 1, n),
        'feature2': rng.normal(0, 1, n),
        'feature3': rng.integers(0, 10, n),
        'target': np.concatenate([np.zeros(475), np.ones(25)]).astype(int),
    })


@pytest.fixture
def missing_df():
    """Dataset with missing values."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'col1': rng.normal(0, 1, 200),
        'col2': rng.normal(0, 1, 200),
        'col3': rng.choice(['A', 'B', 'C', None], 200),
        'target': rng.integers(0, 2, 200),
    })
    # Add ~10% missing in numeric
    missing_idx = rng.choice(200, 20, replace=False)
    df.loc[missing_idx, 'col1'] = np.nan
    return df


@pytest.fixture
def outlier_df():
    """Dataset with extreme outliers — 5 outliers in 200 rows = 2.5% (above the 1% threshold)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'normal': rng.normal(100, 10, 200),
        'outlier_col': rng.normal(0, 1, 200),
        'target': rng.integers(0, 2, 200),
    })
    # 5 extreme outliers = 2.5% of 200 — exceeds both health (1%) and fix (1%) thresholds
    df.loc[0, 'outlier_col'] = 1000.0
    df.loc[1, 'outlier_col'] = -1000.0
    df.loc[2, 'outlier_col'] = 950.0
    df.loc[3, 'outlier_col'] = -900.0
    df.loc[4, 'outlier_col'] = 800.0
    return df


@pytest.fixture
def skewed_df():
    """Dataset with highly skewed features."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'skewed': np.concatenate([np.ones(400), np.random.exponential(10, 100)]),
        'normal': rng.normal(0, 1, 500),
        'target': rng.integers(0, 2, 500),
    })


@pytest.fixture
def text_df():
    """Text classification dataset — 10 unique sentences, each >5 words.

    Uniqueness rule in detect_text_columns: looks_categorical when
    nunique <= max(2, min(20, n * 0.05)).  For 100 rows that threshold is 5.
    Using 10 unique sentences comfortably exceeds it.
    """
    sentences = [
        "This product is absolutely amazing and I highly recommend it",
        "Terrible quality complete waste of money do not buy",
        "Great service the staff were very helpful and professional",
        "Awful experience I will never return to this store again",
        "Perfect item exactly as described very fast shipping too",
        "Mediocre performance nothing special but does the job adequately",
        "Outstanding value for money best purchase I have made",
        "Disappointing results did not meet any of my expectations",
        "Excellent build quality very durable and well worth the price",
        "Poor customer support took weeks to resolve my simple issue",
    ]
    # 100 rows with 10 unique sentences — well above the categorical threshold
    data = []
    for i in range(100):
        data.append({'review': sentences[i % 10], 'sentiment': i % 2})
    return pd.DataFrame(data)


@pytest.fixture
def single_class_df():
    """Dataset with only one target class (edge case)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'x1': rng.normal(0, 1, 100),
        'x2': rng.normal(0, 1, 100),
        'target': np.zeros(100),  # All zeros
    })


@pytest.fixture
def multiclass_df():
    """Multi-class classification dataset (3 classes)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'f1': rng.normal(0, 1, 300),
        'f2': rng.normal(0, 1, 300),
        'f3': rng.integers(0, 10, 300),
        'target': np.repeat([0, 1, 2], 100),
    })


@pytest.fixture
def regression_df():
    """Regression dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 3))
    y = X[:, 0] * 5 + X[:, 1] * 3 + rng.normal(0, 0.5, 300)
    return pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'x3': X[:, 2],
        'target': y,
    })
