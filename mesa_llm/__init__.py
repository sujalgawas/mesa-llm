import datetime

import mesa_llm.tools.inbuilt_tools  # noqa: F401, to register inbuilt tools

from .reasoning.reasoning import Observation, Plan
from .recording.analysis import SimulationAnalyzer, load_and_analyze_simulation
from .recording.integration_hooks import (
    RecordingMixin,
    add_recorder_to_model,
    setup_recording_for_existing_simulation,
)
from .recording.recorder import SimulationEvent, SimulationRecorder
from .tools import ToolManager

__all__ = [
    "Observation",
    "Plan",
    "RecordingMixin",
    "SimulationAnalyzer",
    "SimulationEvent",
    "SimulationRecorder",
    "ToolManager",
    "add_recorder_to_model",
    "load_and_analyze_simulation",
    "setup_recording_for_existing_simulation",
]

__title__ = "Mesa-LLM"
__version__ = "0.0.2"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Project Mesa Team"
