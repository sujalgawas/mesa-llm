import inspect
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


def move_to_location(agent: "LLMAgent", target_coordinates: tuple[float, float]) -> str:
    """Move to a given location in a discrete grid
    Args:
        agent: The agent to move
        target_coordinates: The target coordinates to move to
    Returns:
        A string indicating the agent's new position
    """

    agent.position = target_coordinates

    return f"This agent moved to {target_coordinates}."


def speak_to(
    speaker_agent: "LLMAgent", listener_agent: "LLMAgent", message: str
) -> str:
    """Speak to a given agent by updating the discussion object shared by both agents
    Args:
        speaker_agent: The agent speaking
        listener_agent: The agent listening
        message: The message to speak
    Returns:
        A string indicating the message spoken
    """

    return f"{speaker_agent.id} → {listener_agent.id} : {message}"


# Get all the functions in the module that are not private into a list
inbuilt_tools = [
    obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    if not name.startswith("_")
]
