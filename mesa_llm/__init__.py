import datetime

import mesa_llm.tools.inbuilt_tools  # noqa: F401, to register inbuilt tools

from .reasoning.reasoning import Observation, Plan
from .recording import SimulationEvent, SimulationRecorder
from .recording.integration_hooks import (
    record_model,
)
from .tools import ToolManager

__all__ = [
    "Observation",
    "Plan",
    "SimulationEvent",
    "SimulationRecorder",
    "ToolManager",
    "record_model",
]

__title__ = "Mesa-LLM"
__version__ = "0.0.2"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Project Mesa Team"
