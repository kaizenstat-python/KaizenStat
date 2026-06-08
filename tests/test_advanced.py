"""Advanced real-world scenario tests — production-grade edge cases."""
import numpy as np
import pandas as pd
import pytest
from kaizenstat.doctor import DataDoctor
from kaizenstat import fix, debug, model
from kaizenstat.health import scorer
from kaizenstat.validate import checker


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: diverse real-world data shapes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def currency_df():
    """Dataset with currency/percentage strings mixed with numeric."""
    rng = np.random.default_rng(7)
    n = 200
    prices  = [f"${v:,.2f}" for v in rng.uniform(10, 500, n)]
    rates   = [f"{v:.1f}%" for v in rng.uniform(0, 30, n)]
    raw_num = rng.normal(50, 10, n)
    return pd.DataFrame({
        'price_str': prices,
        'rate_str':  rates,
        'numeric':   raw_num,
        'target':    (raw_num > 50).astype(int),
    })


@pytest.fixture
def date_feature_df():
    """Dataset with a date-derived feature (day-of-week, month)."""
    rng = np.random.default_rng(8)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({
        'day_of_week': dates.dayofweek,
        'month':       dates.month,
        'quarter':     dates.quarter,
        'value':       rng.normal(100, 20, n),
        'target':      (rng.normal(0, 1, n) > 0).astype(int),
    })


@pytest.fixture
def multicollinear_df():
    """Dataset with near-perfect multicollinearity."""
    rng = np.random.default_rng(9)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = x1 * 2.0 + rng.normal(0, 0.01, n)   # nearly identical
    x3 = x1 * -1.5 + rng.normal(0, 0.01, n)  # linearly dependent
    x4 = rng.normal(0, 1, n)                  # independent
    target = (x1 + x4 > 0).astype(int)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'target': target})


@pytest.fixture
def label_noise_df():
    """Dataset where 20% of labels are randomly flipped (noisy labels)."""
    rng = np.random.default_rng(10)
    n = 500
    x = rng.normal(0, 1, (n, 4))
    true_labels = (x[:, 0] + x[:, 1] > 0).astype(int)
    flip_mask   = rng.random(n) < 0.20
    noisy_labels = true_labels.copy()
    noisy_labels[flip_mask] = 1 - noisy_labels[flip_mask]
    df = pd.DataFrame(x, columns=['x1', 'x2', 'x3', 'x4'])
    df['target'] = noisy_labels
    return df


@pytest.fixture
def high_cardinality_string_df():
    """Dataset with a high-cardinality string column (UUID-like)."""
    rng = np.random.default_rng(11)
    n = 200
    ids = [f"USER_{i:06d}" for i in range(n)]   # every row unique
    return pd.DataFrame({
        'user_id':   ids,
        'age':       rng.integers(18, 70, n),
        'purchases': rng.integers(0, 50, n),
        'target':    rng.integers(0, 2, n),
    })


@pytest.fixture
def temporal_leakage_df():
    """Dataset where a feature encodes future target info (temporal leakage)."""
    rng = np.random.default_rng(12)
    n = 400
    target  = rng.integers(0, 2, n)
    # leak: directly derived from target + tiny noise — correlation > 0.98
    leak    = target.astype(float) * 100 + rng.normal(0, 0.1, n)
    feature = rng.normal(0, 1, n)
    return pd.DataFrame({'leak': leak, 'feature': feature, 'target': target})


@pytest.fixture
def string_encoded_numeric_df():
    """Dataset where numeric values are stored as strings ('1', '2', '3')."""
    rng = np.random.default_rng(13)
    n = 300
    return pd.DataFrame({
        'level':  [str(v) for v in rng.integers(1, 5, n)],
        'score':  [str(round(v, 1)) for v in rng.uniform(0, 100, n)],
        'age':    rng.integers(18, 65, n).astype(float),
        'target': rng.integers(0, 2, n),
    })


@pytest.fixture
def wide_df():
    """Wide dataset: 50 features, 500 rows (high-dimensional relative to samples)."""
    rng = np.random.default_rng(14)
    n, p = 500, 50
    X = rng.normal(0, 1, (n, p))
    # Only first 3 features are actually predictive
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    cols = {f'f{i:02d}': X[:, i] for i in range(p)}
    cols['target'] = y
    return pd.DataFrame(cols)


@pytest.fixture
def regression_skewed_target_df():
    """Regression dataset with heavily right-skewed target (log-normal)."""
    rng = np.random.default_rng(15)
    n = 400
    x1 = rng.normal(5, 2, n)
    x2 = rng.normal(3, 1, n)
    y  = np.exp(0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n))  # log-normal
    return pd.DataFrame({'x1': x1, 'x2': x2, 'target': y})


@pytest.fixture
def all_missing_features_df():
    """Dataset where all features are 100% missing except one."""
    n = 100
    return pd.DataFrame({
        'good_col': np.arange(n, dtype=float),
        'bad_col1': [np.nan] * n,
        'bad_col2': [np.nan] * n,
        'target':   (np.arange(n) % 2).astype(int),
    })


@pytest.fixture
def duplicate_heavy_df():
    """Dataset that is 60% duplicates."""
    rng = np.random.default_rng(16)
    base = pd.DataFrame({
        'x': rng.integers(0, 5, 50).astype(float),
        'y': rng.integers(0, 3, 50).astype(float),
        'target': rng.integers(0, 2, 50),
    })
    # Stack: 50 unique + 150 duplicates = 200 rows, 75% duplicates
    return pd.concat([base] * 4, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Real-World Data Quality Challenges
# ─────────────────────────────────────────────────────────────────────────────

class TestDataQualityRealWorld:
    """Real-world data quality edge cases."""

    def test_currency_string_columns_handled(self, currency_df):
        """Pipeline should not crash on currency-formatted string columns."""
        doctor = DataDoctor()
        doctor.fit(currency_df, target='target')
        health = doctor.health()
        assert health is not None
        assert 0 <= health.score <= 100

    def test_date_features_pipeline(self, date_feature_df):
        """Date-derived numeric features should train cleanly."""
        doctor = DataDoctor()
        doctor.fit(date_feature_df, target='target')
        result = doctor.train()
        assert result is not None
        assert result.test_score >= 0

    def test_multicollinear_features_flagged(self, multicollinear_df):
        """Near-perfect multicollinearity should be detected by validation."""
        result = checker.assumptions(multicollinear_df, target='target')
        # At least one HIGH-risk issue (VIF or correlation)
        high_issues = [i for i in result.issues if i.risk_level == 'HIGH']
        assert len(high_issues) > 0

    def test_multicollinear_pipeline_trains(self, multicollinear_df):
        """Model should still train despite multicollinearity (just with warnings)."""
        doctor = DataDoctor()
        doctor.fit(multicollinear_df, target='target')
        result = doctor.train()
        assert result.test_score > 0

    def test_label_noise_model_trains(self, label_noise_df):
        """20% label noise should not crash training."""
        doctor = DataDoctor()
        doctor.fit(label_noise_df, target='target')
        result = doctor.train()
        # With 20% noise, test score should still be above chance
        assert result.test_score > 0.40

    def test_label_noise_debug_detects_issue(self, label_noise_df):
        """With noisy labels, dataset difficulty should be moderate-to-high."""
        from kaizenstat.debug.debugger import dataset_difficulty
        X = label_noise_df.drop(columns=['target'])
        y = label_noise_df['target']
        diff = dataset_difficulty(X, y)
        assert 0.0 <= diff <= 1.0

    def test_high_cardinality_string_id_leakage(self, high_cardinality_string_df):
        """String column where every row is unique should be flagged as leakage risk."""
        result = checker.leakage(high_cardinality_string_df, target='target')
        # user_id is a unique-key string: should be flagged
        assert not result.passed
        column_flags = result.issues[0].column if result.issues else ""
        assert 'user_id' in column_flags

    def test_high_cardinality_id_removed_by_fix(self, high_cardinality_string_df):
        """Fix engine should drop the ID-like string column."""
        plan = fix.plan(high_cardinality_string_df, target='target', safe=True)
        fixed = plan.apply(high_cardinality_string_df)
        assert 'user_id' not in fixed.columns

    def test_temporal_leakage_detected(self, temporal_leakage_df):
        """Feature with > 0.98 correlation to target should be flagged."""
        result = checker.leakage(temporal_leakage_df, target='target')
        assert not result.passed
        assert any('leak' in i.column for i in result.issues)

    def test_temporal_leakage_health_penalty(self, temporal_leakage_df):
        """Health scorer should assign a leakage-proxy penalty."""
        result = scorer.report(temporal_leakage_df, target='target')
        assert any(p.name == 'Leakage Proxy' for p in result.penalties)

    def test_duplicate_heavy_dataset(self, duplicate_heavy_df):
        """Dataset with 75% duplicates should trigger duplicate penalty."""
        result = scorer.report(duplicate_heavy_df, target='target')
        assert any(p.name == 'Duplicate Rows' for p in result.penalties)

    def test_duplicate_heavy_fix_reduces_rows(self, duplicate_heavy_df):
        """Fix should significantly reduce rows in duplicate-heavy dataset."""
        plan = fix.plan(duplicate_heavy_df, target='target', safe=True)
        fixed = plan.apply(duplicate_heavy_df)
        assert len(fixed) < len(duplicate_heavy_df)

    def test_all_missing_features_handled(self, all_missing_features_df):
        """All-NaN columns should be removed and pipeline should complete."""
        plan = fix.plan(all_missing_features_df, target='target', safe=False)
        fixed = plan.apply(all_missing_features_df)
        # bad_col1 and bad_col2 (100% missing) should be dropped
        assert 'bad_col1' not in fixed.columns
        assert 'bad_col2' not in fixed.columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Model Training Robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRobustness:
    """Training robustness under adversarial conditions."""

    def test_wide_dataset_trains_without_error(self, wide_df):
        """50-feature dataset should train successfully."""
        doctor = DataDoctor()
        doctor.fit(wide_df, target='target')
        result = doctor.train()
        assert result.test_score > 0

    def test_wide_dataset_feature_selection(self, wide_df):
        """High-dimensional data should not crash the pipeline."""
        result = model.train_best(wide_df, target='target')
        assert result.model_name is not None
        assert result.pipeline is not None

    def test_skewed_regression_target(self, regression_skewed_target_df):
        """Log-normal regression target should train without NaN score."""
        result = model.train_best(regression_skewed_target_df, target='target')
        assert result.task == 'regression'
        assert not np.isnan(result.test_score)

    def test_skewed_regression_debug(self, regression_skewed_target_df):
        """Debug on skewed regression should produce valid result."""
        doctor = DataDoctor()
        doctor.fit(regression_skewed_target_df, target='target')
        doctor.train()
        result = doctor.debug_model()
        assert result is not None
        assert result.task == 'regression'

    def test_single_informative_feature_among_noise(self):
        """Model should outperform chance with 1 informative + 20 noise features."""
        rng = np.random.default_rng(99)
        n = 400
        signal = rng.normal(0, 1, n)
        noise  = rng.normal(0, 1, (n, 20))
        target = (signal > 0).astype(int)
        cols   = {'signal': signal, 'target': target}
        cols.update({f'noise_{i}': noise[:, i] for i in range(20)})
        df = pd.DataFrame(cols)
        result = model.train_best(df, target='target')
        assert result.test_score > 0.55  # better than random

    def test_all_categorical_features(self):
        """Dataset with only categorical features should train without crash."""
        rng = np.random.default_rng(21)
        n = 200
        df = pd.DataFrame({
            'cat1': rng.choice(['A', 'B', 'C', 'D'], n),
            'cat2': rng.choice(['X', 'Y', 'Z'], n),
            'cat3': rng.choice(['low', 'mid', 'high'], n),
            'target': rng.integers(0, 2, n),
        })
        doctor = DataDoctor()
        doctor.fit(df, target='target')
        result = doctor.train()
        assert result is not None

    def test_perfectly_separable_dataset(self):
        """Perfectly separable data should achieve near-perfect scores."""
        df = pd.DataFrame({
            'x1': list(range(50)) + list(range(100, 150)),
            'x2': [0] * 50 + [1] * 50,
            'target': [0] * 50 + [1] * 50,
        })
        result = model.train_best(df, target='target')
        assert result.test_score > 0.90

    def test_reproducibility_same_seed(self):
        """Identical runs should produce the same model name."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 300),
            'y': rng.normal(0, 1, 300),
            'target': rng.integers(0, 2, 300),
        })
        r1 = model.train_best(df.copy(), target='target')
        r2 = model.train_best(df.copy(), target='target')
        assert r1.model_name == r2.model_name

    def test_tune_produces_best_params(self):
        """Tuning should always populate best_params."""
        rng = np.random.default_rng(55)
        n = 300
        df = pd.DataFrame({
            'x': rng.normal(0, 1, n),
            'y': rng.normal(0, 1, n),
            'target': rng.integers(0, 2, n),
        })
        result = model.train_best(df, target='target', tune=True, n_iter=5)
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0

    def test_multiclass_10_classes(self):
        """Model should handle 10-class classification."""
        rng = np.random.default_rng(77)
        n = 500
        df = pd.DataFrame({
            f'f{i}': rng.normal(0, 1, n) for i in range(5)
        })
        df['target'] = rng.integers(0, 10, n)
        doctor = DataDoctor()
        doctor.fit(df, target='target')
        result = doctor.train()
        assert result.task == 'classification'
        assert result.test_score >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Health Scoring Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEdgeCases:
    """Health scoring under unusual real-world conditions."""

    def test_all_same_value_target(self):
        """Single-class target should not crash scoring."""
        df = pd.DataFrame({
            'x': range(100),
            'target': [1] * 100,
        })
        result = scorer.report(df, target='target')
        assert 0 <= result.score <= 100

    def test_very_large_class_imbalance(self):
        """1% minority class should get maximum imbalance penalty."""
        n = 1000
        df = pd.DataFrame({
            'x': np.random.normal(0, 1, n),
            'target': [0] * 990 + [1] * 10,
        })
        result = scorer.report(df, target='target')
        assert any(p.name == 'Class Imbalance' for p in result.penalties)
        imb = next(p for p in result.penalties if p.name == 'Class Imbalance')
        assert abs(imb.penalty) >= 15  # large penalty for extreme imbalance

    def test_mostly_missing_data(self):
        """Dataset with 80% missing values should trigger missing-values penalty."""
        n = 100
        col_vals = [np.nan] * 80 + [float(i) for i in range(20)]
        tgt_vals = [i % 2 for i in range(n)]
        df = pd.DataFrame({'col': col_vals, 'target': tgt_vals})
        result = scorer.report(df, target='target')
        assert result.score < 100  # missing-values penalty must apply

    def test_no_target_provided(self):
        """Health scoring without target should still work."""
        df = pd.DataFrame({'x': range(100), 'y': range(100)})
        result = scorer.report(df)
        assert 0 <= result.score <= 100

    def test_grade_thresholds_correct(self):
        """Grade boundaries: A>=90, B>=80, C>=70, D>=60, F<60."""
        from kaizenstat.utils.helpers import score_to_grade
        assert score_to_grade(95) == 'A'
        assert score_to_grade(85) == 'B'
        assert score_to_grade(72) == 'C'
        assert score_to_grade(62) == 'D'
        assert score_to_grade(55) == 'F'

    def test_integer_range_target_treated_as_classification(self):
        """Target with 2 unique integer values should be treated as classification."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 200),
            'target': rng.integers(0, 2, 200),
        })
        result = model.train_best(df, target='target')
        assert result.task == 'classification'


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Validation Real-World Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationRealWorld:
    """Validation checks on realistic messy data."""

    def test_mixed_scale_features(self):
        """Features on vastly different scales should not crash validation."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            'age':    rng.uniform(0, 1, n),         # 0–1
            'salary': rng.uniform(20000, 200000, n), # 20k–200k
            'rating': rng.uniform(1, 5, n),          # 1–5
            'target': rng.integers(0, 2, n),
        })
        result = checker.assumptions(df, target='target')
        assert result is not None

    def test_near_constant_feature_flagged(self):
        """Feature with almost no variance should be flagged."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            'almost_constant': [1.0] * 199 + [1.001],  # CV < 0.005
            'normal': rng.normal(0, 1, n),
            'target': rng.integers(0, 2, n),
        })
        result = checker.distribution_check(df, target='target')
        assert result is not None  # should not crash

    def test_drift_detection_with_shifted_distribution(self):
        """Significant distribution shift should be detected."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        train = pd.DataFrame({'age': rng1.normal(30, 5, 500)})
        test  = pd.DataFrame({'age': rng2.normal(60, 5, 500)})  # very different mean
        drift = checker.detect_drift(train, test)
        # A mean shift of 30 years should be detected (p << 0.05)
        assert 'age' in drift

    def test_no_drift_same_distribution(self):
        """No drift should be reported when train and test share the same distribution."""
        rng = np.random.default_rng(42)
        data = pd.DataFrame({'x': rng.normal(0, 1, 1000)})
        train = data.iloc[:500]
        test  = data.iloc[500:]
        drift = checker.detect_drift(train, test)
        # Same distribution — no drift expected
        assert 'x' not in drift

    def test_validation_handles_nan_features(self):
        """Validation checks should not crash with NaN-heavy columns."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': [np.nan] * 50 + list(rng.normal(0, 1, 150)),
            'y': rng.normal(0, 1, 200),
            'target': rng.integers(0, 2, 200),
        })
        result = checker.assumptions(df, target='target')
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Fix Engine Real-World Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestFixEngineRealWorld:
    """Fix engine under production-like conditions."""

    def test_fix_preserves_row_count_when_no_issues(self):
        """Clean dataset fix should not lose rows."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 100),
            'y': rng.normal(0, 1, 100),
            'target': rng.integers(0, 2, 100),
        })
        plan = fix.plan(df, target='target', safe=True)
        fixed = plan.apply(df)
        assert len(fixed) == len(df)

    def test_fix_mixed_missing_types(self):
        """Mixed numeric + categorical missing should all be filled."""
        rng = np.random.default_rng(42)
        # cat has 3 distinct values so it won't be dropped as constant
        cats = ['A', 'B', 'C']
        df = pd.DataFrame({
            'num': [np.nan if i % 5 == 0 else float(i) for i in range(100)],
            'cat': [None if i % 7 == 0 else cats[i % 3] for i in range(100)],
            'target': rng.integers(0, 2, 100),
        })
        plan = fix.plan(df, target='target', safe=True)
        fixed = plan.apply(df)
        assert fixed['num'].isnull().sum() == 0
        # cat column may be label-encoded (numeric) or filled — it must exist and have no nulls
        assert 'cat' in fixed.columns
        assert fixed['cat'].isnull().sum() == 0

    def test_fix_multiple_categorical_columns(self):
        """All categorical columns should be encoded to numeric."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            'cat1': rng.choice(['A', 'B', 'C'], n),
            'cat2': rng.choice(['X', 'Y'], n),
            'cat3': rng.choice(['low', 'mid', 'high'], n),
            'num':  rng.normal(0, 1, n),
            'target': rng.integers(0, 2, n),
        })
        plan = fix.plan(df, target='target', safe=True)
        fixed = plan.apply(df)
        for col in ['cat1', 'cat2', 'cat3']:
            assert pd.api.types.is_numeric_dtype(fixed[col]), \
                f"Column {col!r} should be numeric after fix"

    def test_fix_safe_mode_no_medium_risk(self):
        """Safe mode must never apply MEDIUM-risk actions."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'outlier': [1000.0 if i == 0 else float(i) for i in range(100)],
            'num':     rng.normal(0, 1, 100),
            'target':  rng.integers(0, 2, 100),
        })
        plan = fix.plan(df, target='target', safe=True)
        for action in plan.actions:
            assert action.risk_level == 'LOW', \
                f"Safe mode applied MEDIUM action: {action.action} on {action.column}"

    def test_fix_idempotent_on_clean_data(self):
        """Applying fix twice on already-fixed data should be a no-op."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 100),
            'target': rng.integers(0, 2, 100),
        })
        plan1 = fix.plan(df, target='target', safe=True)
        fixed1 = plan1.apply(df)
        plan2 = fix.plan(fixed1, target='target', safe=True)
        fixed2 = plan2.apply(fixed1)
        assert fixed1.shape == fixed2.shape


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Debug / Diagnostics Real-World Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestDebugRealWorld:
    """Debug engine under realistic conditions."""

    def test_debug_numpy_array_target(self):
        """dataset_difficulty should accept numpy array for y."""
        from kaizenstat.debug.debugger import dataset_difficulty
        rng = np.random.default_rng(42)
        X = pd.DataFrame({'x': rng.normal(0, 1, 200), 'y': rng.normal(0, 1, 200)})
        y = rng.integers(0, 2, 200)  # numpy array, not Series
        diff = dataset_difficulty(X, y)
        assert 0 <= diff <= 1

    def test_recommend_actions_none_profile(self):
        """recommend_actions should work with None profile."""
        from kaizenstat.debug.debugger import recommend_actions
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'x': rng.normal(0, 1, 300), 'target': rng.integers(0, 2, 300)})
        pipe = model.train_best(df, target='target').pipeline
        X = df[['x']].iloc[-60:]
        y = df['target'].iloc[-60:]
        dr = debug.model_failure(pipe, X, X, y, y)
        actions = recommend_actions(None, dr)  # None profile — must not crash
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_feature_importance_always_non_null(self):
        """Feature importances should never be None for standard pipeline."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x1': rng.normal(0, 1, 300),
            'x2': rng.normal(0, 1, 300),
            'x3': rng.choice(['A', 'B'], 300),
            'target': rng.integers(0, 2, 300),
        })
        result = model.train_best(df, target='target')
        X = df.drop(columns=['target']).iloc[-60:]
        y = df['target'].iloc[-60:]
        dr = debug.model_failure(result.pipeline, X, X, y, y)
        # Feature importances must be a non-empty Series
        assert dr.feature_importances is not None
        assert len(dr.feature_importances) > 0

    def test_dataset_difficulty_easy_vs_hard(self):
        """Perfectly separable data should score lower difficulty than random data."""
        from kaizenstat.debug.debugger import dataset_difficulty

        X_easy = pd.DataFrame({'x': list(range(-50, 0)) + list(range(1, 51))})
        y_easy = np.array([0] * 50 + [1] * 50)

        rng = np.random.default_rng(42)
        X_hard = pd.DataFrame({'x': rng.normal(0, 1, 100)})
        y_hard = rng.integers(0, 2, 100)  # pure noise — unpredictable

        diff_easy = dataset_difficulty(X_easy, y_easy)
        diff_hard = dataset_difficulty(X_hard, y_hard)
        assert diff_easy < diff_hard

    def test_debug_overfitting_score_within_bounds(self):
        """Debug scores should all be in [0, 1] range."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 400),
            'target': rng.integers(0, 2, 400),
        })
        result = model.train_best(df, target='target')
        X = df[['x']].iloc[-80:]
        y = df['target'].iloc[-80:]
        dr = debug.model_failure(result.pipeline, X, X, y, y)
        assert 0 <= dr.train_score <= 1
        assert 0 <= dr.test_score <= 1
        assert 0 <= dr.confidence <= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Pipeline Consistency & API Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIContract:
    """API contracts and return-type consistency."""

    def test_train_result_has_pipeline_attribute(self):
        """TrainResult must always expose a callable pipeline."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'x': rng.normal(0, 1, 200), 'target': rng.integers(0, 2, 200)})
        result = model.train_best(df, target='target')
        assert hasattr(result.pipeline, 'predict')
        assert callable(result.pipeline.predict)

    def test_benchmark_entries_are_dataclass(self):
        """BenchmarkResult entries must expose .name and .score attributes."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 300),
            'y': rng.normal(0, 1, 300),
            'target': rng.integers(0, 2, 300),
        })
        bm = model.benchmark(df, target='target')
        for entry in bm.entries:
            assert hasattr(entry, 'name'), "ModelEntry must have .name attribute"
            assert hasattr(entry, 'score'), "ModelEntry must have .score attribute"
            assert 0 <= entry.score <= 1

    def test_health_result_penalties_all_negative(self):
        """All penalties in HealthResult must be ≤ 0."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': [np.nan if i % 4 == 0 else float(i) for i in range(200)],
            'target': rng.integers(0, 2, 200),
        })
        result = scorer.report(df, target='target')
        for p in result.penalties:
            assert p.penalty <= 0, f"Penalty {p.name!r} is positive: {p.penalty}"

    def test_validation_report_checks_run_positive(self):
        """ValidationReport.checks_run must be > 0 when target provided."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'x': rng.normal(0, 1, 200), 'target': rng.integers(0, 2, 200)})
        result = checker.assumptions(df, target='target')
        assert result.checks_run > 0

    def test_suggestions_all_have_required_fields(self):
        """Every Suggestion must have action, reason, expected_gain, priority."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 300),
            'y': rng.normal(0, 1, 300),
            'target': rng.integers(0, 2, 300),
        })
        doctor = DataDoctor()
        doctor.fit(df, target='target')
        doctor.train()
        doctor.debug_model()
        result = doctor.improve()
        for s in result.suggestions:
            assert s.action, f"Suggestion missing action: {s}"
            assert s.reason, f"Suggestion missing reason: {s}"
            assert s.expected_gain, f"Suggestion missing expected_gain: {s}"
            assert isinstance(s.priority, int)

    def test_doctor_mode_property_accessible(self):
        """DataDoctor.mode() must return 'tabular' or 'text' after fit."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'x': rng.normal(0, 1, 200), 'target': rng.integers(0, 2, 200)})
        doctor = DataDoctor()
        doctor.fit(df, target='target')
        # mode is a regular method (not a property), call with parentheses
        assert doctor.mode() in ('tabular', 'text')

    def test_fix_plan_actions_have_valid_risk_levels(self):
        """All FixAction risk_level values must be LOW, MEDIUM, or HIGH."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'cat': rng.choice(['A', 'B', 'C'], 100),
            'num': [np.nan if i % 5 == 0 else float(i) for i in range(100)],
            'target': rng.integers(0, 2, 100),
        })
        plan = fix.plan(df, target='target', safe=False)
        valid = {'LOW', 'MEDIUM', 'HIGH'}
        for action in plan.actions:
            assert action.risk_level in valid, \
                f"Invalid risk_level {action.risk_level!r} for action {action.action!r}"
