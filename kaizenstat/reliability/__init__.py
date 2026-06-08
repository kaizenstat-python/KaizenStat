"""
reliability — Production-readiness Trust & Reliability layer.

Usage::

    from kaizenstat import reliability

    reliability.analyze(model, X_test, y_test)   # → TrustReport

Or via DataDoctor::

    doctor.trust_score()   # → TrustReport (production-readiness verdict)
"""
from .trust import TrustAnalyzer, TrustReport, analyze

__all__ = ["TrustAnalyzer", "TrustReport", "analyze"]
