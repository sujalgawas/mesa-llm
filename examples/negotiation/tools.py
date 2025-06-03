from typing import TYPE_CHECKING

from examples.negotiation.agents import buyer_tool_manager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=buyer_tool_manager)
def set_chosen_brand(agent: "LLMAgent", chosen_brand: str) -> str:
    """
    A tool to set the brand of choice of the buyer agent, It can either be brand A or brand B.

    Args:
        agent : The agent to set the brand of choice for.
        chosen_brand : The brand of choice of the buyer agent, either "A" or "B".

    Returns:
        str: The brand of choice of the buyer agent, either "A" or "B".
    """
    agent.chosen_brand = chosen_brand
    return f"Chosen brand of {agent} set to {chosen_brand}"
