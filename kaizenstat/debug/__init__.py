"""
debug — ML Model Debugger module.

Usage::

    from kaizenstat import debug

    debug.model_failure(model, X_train, X_test, y_train, y_test)
    debug.overfitting_check(model, X_train, X_test, y_train, y_test)
    debug.error_analysis(model, X_test, y_test)
    debug.feature_importance(model, X_test, y_test)
    debug.bias_detection(model, X_test, y_test, sensitive_features=["gender"])
"""
from .debugger import (
    ModelDebugger,
    DebugResult,
    DebugIssue,
    model_failure,
    overfitting_check,
    error_analysis,
    feature_importance,
    feature_impact,
    dataset_difficulty,
    recommend_actions,
    bias_detection,
)
from . import text_debugger
from .text_debugger import TextModelDebugger

__all__ = [
    "ModelDebugger",
    "DebugResult",
    "DebugIssue",
    "model_failure",
    "overfitting_check",
    "error_analysis",
    "feature_importance",
    "feature_impact",
    "dataset_difficulty",
    "recommend_actions",
    "bias_detection",
    "text_debugger",
    "TextModelDebugger",
]
