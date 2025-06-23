from typing import TYPE_CHECKING

from examples.epstein_civil_violence.agents import (
    citizen_tool_manager,
    cop_tool_manager,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=citizen_tool_manager)
def move_one_step(agent: "LLMAgent") -> str:
    print("moved")


# @tool(tool_manager=cop_tool_manager)
# def move_one_step(agent: "LLMAgent") -> str:
#     print("moved")


@tool(tool_manager=citizen_tool_manager)
def join_the_rebels(agent: "LLMAgent") -> str:
    print("joined the rebels")


@tool(tool_manager=cop_tool_manager)
def arrest_citizen(agent: "LLMAgent", citizen_id: str) -> str:
    print("arrested citizens")
