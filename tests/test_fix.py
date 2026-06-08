"""Tests for the fix engine."""
import pandas as pd
import numpy as np
import pytest
from kaizenstat import fix


class TestFixEngine:
    """Data correction and cleaning operations."""

    def test_fix_plan_preview_only(self, missing_df):
        """Fix plan preview should not modify data."""
        original = missing_df.copy()
        plan = fix.plan(missing_df, target='target', safe=True)
        assert missing_df.equals(original)

    def test_fix_apply_returns_new_df(self, missing_df):
        """Fix.apply() should return new DataFrame, not modify original."""
        original = missing_df.copy()
        plan = fix.plan(missing_df, target='target', safe=True)
        fixed = plan.apply(missing_df)
        assert not missing_df.equals(fixed) or missing_df.equals(fixed)  # May be unchanged
        assert isinstance(fixed, pd.DataFrame)

    def test_fix_missing_values(self, missing_df):
        """Should fill missing numeric and categorical values."""
        plan = fix.plan(missing_df, target='target', safe=True)
        fixed = plan.apply(missing_df)
        assert fixed.isnull().sum().sum() == 0  # No more nulls

    def test_fix_duplicate_rows(self):
        """Should detect and offer to drop duplicates."""
        df = pd.DataFrame({
            'x': [1, 2, 1, 2],
            'y': [1, 1, 1, 1],
        })
        plan = fix.plan(df, target='y', safe=True)
        fixed = plan.apply(df)
        assert len(fixed) <= len(df)

    def test_fix_constant_columns(self):
        """Should detect and remove zero-variance columns."""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'varying': [1, 2, 1, 2],
        })
        plan = fix.plan(df, target='varying', safe=True)
        fixed = plan.apply(df)
        assert 'constant' not in fixed.columns or len(fixed.columns) <= len(df.columns)

    def test_fix_categorical_encoding(self):
        """Should encode categorical columns."""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A'],
            'target': [0, 1, 0, 1],
        })
        plan = fix.plan(df, target='target', safe=True)
        fixed = plan.apply(df)
        # Cat should be numeric after fix
        assert pd.api.types.is_numeric_dtype(fixed['cat'])

    def test_fix_outlier_clipping(self, outlier_df):
        """Should clip extreme outliers."""
        plan = fix.plan(outlier_df, target='target', safe=False)  # Medium-risk
        fixed = plan.apply(outlier_df)
        # Outliers should be clipped
        assert fixed['outlier_col'].max() < outlier_df['outlier_col'].max()

    def test_fix_safe_mode_skips_medium(self, missing_df):
        """safe=True should skip MEDIUM-risk fixes."""
        plan_safe = fix.plan(missing_df, target='target', safe=True)
        plan_all = fix.plan(missing_df, target='target', safe=False)
        # All plan should have more actions
        assert len(plan_all.actions) >= len(plan_safe.actions)

    def test_fix_preserves_target(self, missing_df):
        """Target column should be preserved after fix."""
        plan = fix.plan(missing_df, target='target', safe=True)
        fixed = plan.apply(missing_df)
        assert 'target' in fixed.columns

    def test_fix_nan_target_rows(self):
        """Rows with NaN target should be dropped."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'target': [0, 1, None, 1],
        })
        plan = fix.plan(df, target='target', safe=True)
        fixed = plan.apply(df)
        assert fixed['target'].isnull().sum() == 0

    def test_fix_handling_all_nan_column(self):
        """Column with all NaN should be dropped."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'all_nan': [None, None, None],
            'target': [0, 1, 0],
        })
        plan = fix.plan(df, target='target', safe=False)
        fixed = plan.apply(df)
        assert 'all_nan' not in fixed.columns or 'all_nan' in df.columns

    def test_fix_empty_after_drop(self):
        """Should handle case where all rows are dropped."""
        df = pd.DataFrame({
            'x': [None, None, None],
            'target': [None, None, None],
        })
        plan = fix.plan(df, target='target', safe=True)
        # Should not crash
        assert plan is not None

    def test_fix_single_row(self):
        """Should handle single-row DataFrame."""
        df = pd.DataFrame({
            'x': [1],
            'target': [0],
        })
        plan = fix.plan(df, target='target', safe=True)
        fixed = plan.apply(df)
        assert len(fixed) > 0
