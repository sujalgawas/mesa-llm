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
    agent: "LLMAgent", target_coordinates: tuple[float, float] | list[float]
) -> str:
    """
    Teleport to a given location in a grid or continuous space.

    Args:
        agent: The agent to move (as a LLM, ignore this argument in function calling).
        target_coordinates: The target coordinates to move to, specified as a tuple of (x, y) floats.

    Returns:
        A string indicating the agent's new position.
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
        agent.model.get_agent(listener_agent_unique_id)
        for listener_agent_unique_id in listener_agents_unique_ids
    ]

    for recipient in [*listener_agents, agent]:
        recipient.memory.add_to_memory(
            type="Message",
            content=message,
            step=agent.model.steps,
            metadata={
                "sender": agent,
                "recipients": listener_agents,
            },
        )
    return f"{agent.unique_id} → {listener_agents} : {message}"


if __name__ == "__main__":
    # CL to execute this file: python -m mesa_llm.tools.inbuilt_tools
    import json

    print(json.dumps(teleport_to_location.__tool_schema__, indent=2))
    print(json.dumps(speak_to.__tool_schema__, indent=2))
