"""
model — Model benchmarking and training module.

Usage::

    from kaizenstat import model

    result = model.benchmark(df, target="y")
    result = model.train_best(df, target="y")
    metrics = model.evaluate(pipeline, X_test, y_test)
"""
from .trainer import (
    ModelTrainer,
    BenchmarkResult,
    TrainResult,
    ModelEntry,
    benchmark,
    train_best,
    train_auto,
    evaluate,
)
from . import text_trainer
from .text_trainer import TextModelTrainer

__all__ = [
    "ModelTrainer",
    "BenchmarkResult",
    "TrainResult",
    "ModelEntry",
    "benchmark",
    "train_best",
    "train_auto",
    "evaluate",
    "text_trainer",
    "TextModelTrainer",
]
