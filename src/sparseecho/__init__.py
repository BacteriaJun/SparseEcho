"""SparseEcho SFPTI reconstruction runtime."""

from .calibration import CalibrationSnapshot, JsonCalibrationStore, StaticCalibrationProvider
from .config import EngineConfig, SceneConfig
from .engine import SparseEchoEngine
from .planning import (
    AcquisitionDecision,
    AcquisitionRequest,
    AdaptiveAcquisitionController,
    ApertureBudget,
    QueryPlan,
    compile_query_plan,
)
from .runtime import (
    CaptureDirectoryBackend,
    CaptureFrame,
    JsonlTelemetrySink,
    ReconstructionRuntime,
    RuntimeFault,
    RuntimeLimits,
    RuntimeOutcome,
    RuntimeState,
)
from .session import AdaptiveReconstructionSession, SessionResult

__version__ = "1.1"

__all__ = [
    "AcquisitionDecision",
    "AcquisitionRequest",
    "AdaptiveAcquisitionController",
    "AdaptiveReconstructionSession",
    "ApertureBudget",
    "CalibrationSnapshot",
    "CaptureDirectoryBackend",
    "CaptureFrame",
    "EngineConfig",
    "JsonCalibrationStore",
    "JsonlTelemetrySink",
    "QueryPlan",
    "ReconstructionRuntime",
    "RuntimeFault",
    "RuntimeLimits",
    "RuntimeOutcome",
    "RuntimeState",
    "SceneConfig",
    "SessionResult",
    "SparseEchoEngine",
    "StaticCalibrationProvider",
    "compile_query_plan",
    "__version__",
]
