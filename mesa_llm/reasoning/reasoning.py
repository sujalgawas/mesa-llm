from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        pass

    async def aplan(
        self,
        prompt: str,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        Default implementation calls the synchronous plan() method.
        """
        return self.plan(prompt, obs, ttl, selected_tools)

    def execute_tool_call(
        self, chaining_message, selected_tools: list[str] | None = None
    ):
        system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
        self.agent.llm.system_prompt = system_prompt
        rsp = self.agent.llm.generate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(
                selected_tools=selected_tools
            ),
            tool_choice="required",
        )
        response_message = rsp.choices[0].message
        plan = Plan(step=self.agent.model.steps, llm_plan=response_message, ttl=1)

        return plan

    async def aexecute_tool_call(
        self, chaining_message, selected_tools: list[str] | None = None
    ):
        """
        Asynchronous version of execute_tool_call() method.
        """
        system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
        self.agent.llm.system_prompt = system_prompt
        rsp = await self.agent.llm.agenerate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(
                selected_tools=selected_tools
            ),
            tool_choice="required",
        )
        response_message = rsp.choices[0].message
        plan = Plan(step=self.agent.model.steps, llm_plan=response_message, ttl=1)

        return plan
