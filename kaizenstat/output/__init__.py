"""
output — Reporting and export module.

Usage::

    from kaizenstat import output

    output.summary(health_result=hr, debug_result=dr)
    output.html(results, path="report.html")
    output.export_model(pipeline, path="model.joblib")
    output.codegen(df, target="y")
"""
from .reporter import Reporter, summary, html, export_model, load_model, codegen

__all__ = ["Reporter", "summary", "html", "export_model", "load_model", "codegen"]
