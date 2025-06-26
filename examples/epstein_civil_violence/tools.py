import random
from typing import TYPE_CHECKING

from examples.epstein_civil_violence.agents import (
    CitizenState,
    citizen_tool_manager,
    cop_tool_manager,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


# Mapping directions to (dx, dy)
direction_map = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
    "NorthEast": (1, 1),
    "NorthWest": (-1, 1),
    "SouthEast": (1, -1),
    "SouthWest": (-1, -1),
}

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


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
    citizen.jail_senttence_left = random.randint(1, agent.max_jail_term)
    return f"agent {citizen.unique_id} arrested by {agent.unique_id}."


@tool(tool_manager=citizen_tool_manager)
def move_citizen(agent: "LLMAgent", direction: str) -> str:
    """
    Move the agent one step in the specified direction.

        Args:
            direction: The direction to move in. Must be one of:
                'North', 'South', 'East', 'West',
                'NorthEast', 'NorthWest', 'SouthEast', or 'SouthWest'.
            agent: Provided automatically.

        Returns:
            A string confirming the result of the movement attempt.
    """
    dx, dy = direction_map[direction]
    x, y = agent.pos
    new_pos = (x + dx, y + dy)

    # Make sure the new position is within grid bounds
    if agent.model.grid.out_of_bounds(new_pos):
        return f"Can't move {direction}, out of bounds."
    agent.model.grid.move_agent(agent, new_pos)


@tool(tool_manager=cop_tool_manager)
def move_cop(agent: "LLMAgent", direction: str) -> str:
    """
    Move the agent one step in the specified direction.

        Args:
            direction: The direction to move in. Must be one of:
                'North', 'South', 'East', 'West',
                'NorthEast', 'NorthWest', 'SouthEast', or 'SouthWest'.
            agent: Provided automatically.

        Returns:
            A string confirming the result of the movement attempt.
    """
    dx, dy = direction_map[direction]
    x, y = agent.pos
    new_pos = (x + dx, y + dy)

    # Make sure the new position is within grid bounds
    if agent.model.grid.out_of_bounds(new_pos):
        return f"Can't move {direction}, out of bounds."
    agent.model.grid.move_agent(agent, new_pos)
