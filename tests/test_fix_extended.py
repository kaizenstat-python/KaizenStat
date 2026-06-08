"""Extended fix/engine.py tests — action functions, missing(), encoding(), imbalance()."""
import numpy as np
import pandas as pd
import pytest

from kaizenstat.fix.engine import FixEngine
from kaizenstat.fix import engine as fix_mod


@pytest.fixture
def base_df():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(20, 65, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "city": rng.choice(["NY", "LA", "Chicago"], n),
        "target": rng.integers(0, 2, n),
    })
    df.loc[0:9, "age"] = np.nan
    df.loc[0:4, "city"] = None
    return df


@pytest.fixture
def imbalanced_df_fixture():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 500),
        "target": np.concatenate([np.zeros(475), np.ones(25)]).astype(int),
    })


# ── missing() method ─────────────────────────────────────────────────────────

class TestMissingMethod:
    def test_missing_returns_plan(self, base_df):
        plan = FixEngine().missing(base_df, target="target")
        assert plan is not None

    def test_missing_plan_applies(self, base_df):
        engine = FixEngine()
        plan = engine.missing(base_df, target="target")
        fixed = plan.apply(base_df)
        assert isinstance(fixed, pd.DataFrame)


# ── outlier_handling() method ─────────────────────────────────────────────────

class TestOutlierHandling:
    def test_outlier_handling_plan(self, outlier_df):
        plan = FixEngine().outlier_handling(outlier_df, target="target")
        assert plan is not None

    def test_outlier_handling_applies(self, outlier_df):
        engine = FixEngine()
        plan = engine.outlier_handling(outlier_df, target="target")
        fixed = plan.apply(outlier_df)
        assert isinstance(fixed, pd.DataFrame)


# ── encoding() method ─────────────────────────────────────────────────────────

class TestEncodingMethod:
    def test_encoding_returns_plan(self, base_df):
        plan = FixEngine().encoding(base_df, target="target")
        assert plan is not None

    def test_encoding_applies_label_encode(self, base_df):
        plan = FixEngine().encoding(base_df, target="target")
        fixed = plan.apply(base_df)
        # City column should now be numeric
        assert pd.api.types.is_numeric_dtype(fixed["city"])


# ── imbalance() method ────────────────────────────────────────────────────────

class TestImbalanceMethod:
    def test_imbalance_balanced(self, small_df):
        result = FixEngine().imbalance(small_df, target="churn")
        assert isinstance(result, str)

    def test_imbalance_detected(self, imbalanced_df_fixture):
        result = FixEngine().imbalance(imbalanced_df_fixture, target="target")
        assert "imbalance" in result.lower() or "minority" in result.lower()

    def test_imbalance_single_class(self):
        df = pd.DataFrame({"x": range(50), "target": np.zeros(50)})
        result = FixEngine().imbalance(df, target="target")
        assert "Single class" in result

    def test_imbalance_empty_target(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "target": [np.nan, np.nan]})
        result = FixEngine().imbalance(df, target="target")
        assert "No data" in result


# ── apply_action function coverage ───────────────────────────────────────────

class TestApplyActionCoverage:
    """Test all action types via plan.apply() so _apply_action is exercised."""

    def test_fill_mean(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        action = FixAction(column="x", action="fill_mean", reason="test",
                           risk_level="LOW", expected_impact="fill")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert fixed["x"].isna().sum() == 0

    def test_fill_mode(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"cat": ["A", "A", None, "B"]})
        action = FixAction(column="cat", action="fill_mode", reason="test",
                           risk_level="LOW", expected_impact="fill")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert fixed["cat"].isna().sum() == 0

    def test_fill_constant(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        action = FixAction(column="x", action="fill_constant", reason="test",
                           risk_level="LOW", expected_impact="fill",
                           params={"value": -999})
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert -999 in fixed["x"].values

    def test_log1p_transform(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0]})
        action = FixAction(column="x", action="log1p_transform", reason="test",
                           risk_level="LOW", expected_impact="transform")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert fixed["x"].max() < 10.0  # log1p reduces values

    def test_label_encode(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
        action = FixAction(column="cat", action="label_encode", reason="test",
                           risk_level="LOW", expected_impact="encode")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert pd.api.types.is_numeric_dtype(fixed["cat"])

    def test_drop_rows_missing_target(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, np.nan, 1]})
        action = FixAction(column="target", action="drop_rows_missing_target",
                           reason="test", risk_level="LOW", expected_impact="drop")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert len(fixed) == 2

    def test_fill_median(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0]})
        action = FixAction(column="x", action="fill_median", reason="test",
                           risk_level="LOW", expected_impact="fill")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert fixed["x"].isna().sum() == 0

    def test_unknown_action_returns_unchanged(self):
        from kaizenstat.fix.engine import FixAction, FixPlan
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        action = FixAction(column="x", action="unknown_action", reason="test",
                           risk_level="LOW", expected_impact="noop")
        plan = FixPlan(actions=[action], safe=True)
        fixed = plan.apply(df)
        assert list(fixed["x"]) == [1.0, 2.0, 3.0]


# ── module-level API ──────────────────────────────────────────────────────────

class TestFixModuleAPI:
    def test_module_plan(self, small_df):
        plan = fix_mod.plan(small_df, target="churn")
        assert plan is not None

    def test_module_apply(self, small_df):
        result = fix_mod.apply(small_df, target="churn")
        assert isinstance(result, pd.DataFrame)

    def test_module_plan_safe_false(self, base_df):
        plan = fix_mod.plan(base_df, target="target", safe=False)
        assert plan is not None
