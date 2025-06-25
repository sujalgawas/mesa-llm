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

from examples.epstein_civil_violence.agents import (
    CitizenState,
    citizen_tool_manager,
    cop_tool_manager,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=citizen_tool_manager)
def move_one_step(agent: "LLMAgent", target_coordinates: list[int]) -> str:
    """
    Move the agent to specific (x, y) coordinates within the grid.

        Args:
            target_coordinates: Exactly two integers in the form [x, y] that will be used to move the agent in a nearby cell. Example: [3, 7]
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


@tool(tool_manager=citizen_tool_manager)
def change_state(agent: "LLMAgent", state: CitizenState) -> str:
    """
    Change the state of the agent. The state can be one of the following:
    - CitizenState.QUIET
    - CitizenState.ACTIVE
    - CitizenState.ARRESTED

        Args:
            state: The state to change the agent to.
            agent: Provided automatically

        Returns:
            a string confirming the agent's new state.
    """
    agent.state = state
    return f"agent {agent.unique_id} changed state to {state}."


@tool(tool_manager=cop_tool_manager)
def arrest_citizen(agent: "LLMAgent", citizen: "LLMAgent") -> str:
    """
    Arrest a citizen.

        Args:
            citizen: The citizen to arrest.
            agent: Provided automatically

        Returns:
            a string confirming the citizen's arrest.
    """
    citizen.citizen_state = CitizenState.ARRESTED
    return f"agent {citizen.unique_id} arrested by {agent.unique_id}."
