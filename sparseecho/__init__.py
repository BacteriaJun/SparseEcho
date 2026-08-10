"""SparseEcho — Spectrally Factorized Physical-Time Inversion."""

from .config import EngineConfig, SceneConfig
from .engine import SparseEchoEngine
from .planning import ApertureBudget, QueryPlan, compile_query_plan

__version__ = "1.0"

__all__ = [
    "ApertureBudget",
    "EngineConfig",
    "QueryPlan",
    "SceneConfig",
    "SparseEchoEngine",
    "compile_query_plan",
    "__version__",
]
