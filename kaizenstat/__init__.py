"""
KaizenStat — The standard framework for Data Health Measurement and ML Model Debugging.

Clean imports::

    from kaizenstat import DataDoctor
    from kaizenstat import health, validate, fix, debug, model, improve, output, intelligence

Primary interface::

    doctor = DataDoctor()
    doctor.fit(df, target="target")
    doctor.health()
    doctor.validate()
    doctor.fix(safe=True)
    doctor.train()
    doctor.debug_model()
    doctor.improve()
    doctor.report()
"""

__version__ = "0.5.8"
__author__ = "Masuddar Rahman"

# Primary interface
from kaizenstat.doctor.data_doctor import DataDoctor

# Module-level APIs  (from kaizenstat import health; health.score(df))
from kaizenstat import health
from kaizenstat import validate
from kaizenstat import fix
from kaizenstat import model
from kaizenstat import debug
from kaizenstat import improve
from kaizenstat import output
from kaizenstat import intelligence
from kaizenstat import reliability

# Backward compatibility with v0.2.x
from kaizenstat.core import KaizenStat, DataEngine, detect_device

__all__ = [
    # v0.3+ API
    "DataDoctor",
    "health",
    "validate",
    "fix",
    "model",
    "debug",
    "improve",
    "output",
    "intelligence",
    "reliability",
    # v0.2 legacy
    "KaizenStat",
    "DataEngine",
    "detect_device",
    "__version__",
]
