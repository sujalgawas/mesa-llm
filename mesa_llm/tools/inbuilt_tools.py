from typing import TYPE_CHECKING

from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool
def teleport_to_location(
    agent: "LLMAgent", target_coordinates: tuple[float, float]
) -> str:
    """
    Teleport to a given location in a grid or continuous space.

    Args:
        agent: The agent to move.
        target_coordinates: The target coordinates to move to, specified as a tuple of (x, y) floats.

    Returns:
        A string indicating the agent's new position.
    """

    if isinstance(agent.model.grid, SingleGrid | MultiGrid):
        agent.model.grid.move_agent(agent, target_coordinates)

    elif isinstance(agent.model.grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        cell = agent.model.grid._cells[target_coordinates]
        agent.cell = cell

    elif isinstance(agent.model.space, ContinuousSpace):
        agent.model.space.move_agent(agent.model.space, agent, target_coordinates)

    return f"This agent moved to {target_coordinates}."


@tool
def speak_to(
    speaker_agent: "LLMAgent", listener_agents: list["LLMAgent"], message: str
) -> str:
    """
    Send a message to the recipients and commits it to their memory.
    Args:
        speaker_agent: The agent sending the message
        listener_agents: The agents receiving the message
        message: The message to send
    """
    for recipient in [*listener_agents, speaker_agent]:
        recipient.memory.add_to_memory(
            type="Message",
            content=message,
            step=speaker_agent.model.steps,
            metadata={
                "sender": speaker_agent,
                "recipients": listener_agents,
            },
        )
    return f"{speaker_agent} → {listener_agents} : {message}"
