"""
intelligence — Optional AI advisor using Anthropic Claude.

Usage::

    from kaizenstat import intelligence

    intelligence.init(api_key="sk-ant-...")
    intelligence.advise(health_result=hr, debug_result=dr)
    intelligence.ask("Why is my model overfitting?")
"""
from .ai_advisor import AIAdvisor, init, advise, ask

__all__ = ["AIAdvisor", "init", "advise", "ask"]
