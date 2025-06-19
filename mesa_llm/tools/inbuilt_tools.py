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
    agent: "LLMAgent",
    target_coordinates: list[int],
) -> str:
    """
    Teleport the agent to specific (x, y) coordinates within the grid.

    Args:
        target_coordinates: Exactly two integers in the form [x, y] that fall inside the current environment bounds. Example: [3, 7]
        agent: Provided automatically

    Returns:
        a string confirming the agent's new position.

    """
    target_coordinates = tuple(target_coordinates)
    if isinstance(agent.model.grid, SingleGrid | MultiGrid):
        agent.model.grid.move_agent(agent, target_coordinates)

    elif isinstance(agent.model.grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        cell = agent.model.grid._cells[target_coordinates]
        agent.cell = cell

    elif isinstance(agent.model.space, ContinuousSpace):
        agent.model.space.move_agent(agent.model.space, agent, target_coordinates)

    return f"agent {agent.unique_id} moved to {target_coordinates}."


@tool
def speak_to(
    agent: "LLMAgent", listener_agents_unique_ids: list[int], message: str
) -> str:
    """
    Send a message to the recipients and commits it to their memory.
    Args:
        agent: The agent sending the message (as a LLM, ignore this argument in function calling).
        listener_agents_unique_ids: The unique ids of the agents receiving the message
        message: The message to send
    """
    listener_agents = [
        listener_agent
        for listener_agent in agent.model.agents
        if listener_agent.unique_id in listener_agents_unique_ids
        and listener_agent.unique_id != agent.unique_id
    ]

    for recipient in listener_agents:
        recipient.memory.add_to_memory(
            type="message",
            content={
                "message": message,
                "sender": agent.unique_id,
                "recipients": [
                    listener_agent.unique_id for listener_agent in listener_agents
                ],
            },
        )
    return f"{agent.unique_id} → {[agent.unique_id for agent in listener_agents]} : {message}"
