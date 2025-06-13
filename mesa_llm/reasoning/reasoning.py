from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from terminal_style import style

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class Observation:
    """
    Snapshot of everything the agent can see in this step.

    Attributes:
        step (int): The current simulation time step when the observation is made.

        self_state (dict): A dictionary containing comprehensive information about the observing agent itself.
            This includes:
            - System prompt or role-specific context for LLM reasoning (if used)
            - Internal state such as morale, fear, aggression, fatigue, etc (behavioural).
            - Agent's current location or spatial coordinates
            - Any other agent-specific metadata that could influence decision-making

        local_state (dict): A dictionary summarizing the state of nearby agents (within the vision radius).
            - A dictionary of neighboring agents, where each key is the "angent's class name + id" and the value is a dictionary containing the following:
            - position of neighbors
            - Internal state or attributes of neighboring agents

    """

    step: int
    self_state: dict
    local_state: dict

    def __str__(self) -> str:
        lines = [
            # f"Step: {self.step}",
            f"\n {style('└──', color='green')} {style('[Self State]', color='cyan', bold=True)}",
        ]
        for k, v in self.self_state.items():
            lines.append(f"   • {style(k, color='bold')}: {v}")

        lines.append(
            f"\n  {style('└──', color='green')} {style('[Local State of Nearby Agents]', color='cyan', bold=True)}"
        )
        for agent_id, agent_info in self.local_state.items():
            lines.append(f"   • {style(agent_id, color='bold')}:")
            for k, v in agent_info.items():
                lines.append(f"    • {style(k, color='bold')}: {v}")

        return "\n".join(lines)


@dataclass
class Plan:
    """LLM-generated plan that can span ≥1 steps."""

    step: int  # step when the plan was generated
    llm_plan: Any  # complete LLM response message object (contains both content and tool_calls)
    ttl: int = 1  # steps until planning again (ReWOO sets >1)

    def __str__(self) -> str:
        # Extract content from the message object for display
        if hasattr(self.llm_plan, "content") and self.llm_plan.content:
            llm_plan_str = str(self.llm_plan.content).strip()
        else:
            llm_plan_str = str(self.llm_plan).strip()
        return f"{llm_plan_str}\n"


class Reasoning(ABC):
    def __init__(self, agent: "LLMAgent"):
        self.agent = agent

    @abstractmethod
    def plan(
        self,
        prompt: str,
        obs: Observation,
        ttl: int = 1,
    ) -> Plan:
        pass
