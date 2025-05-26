from mesa.agent import Agent
from mesa.model import Model

from mesa_llm import Plan
from mesa_llm.memory import Memory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning import Reasoning
from mesa_llm.tools.tool_manager import ToolManager


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        api_key (str): The API key for the LLM provider.
        model (str): The model to use for the LLM in the format 'provider/model'. Defaults to 'openai/gpt-4o'.
        system_prompt (str | None): Optional system prompt to be used in LLM completions.
        memory (Memory | None): Optional memory instance to attach to this agent. Can only be set once.
        reasoning (str): Optional reasoning method to be used in LLM completions.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.

    """

    def __init__(
        self,
        model: Model,
        api_key: str,
        reasoning: type[Reasoning],
        llm_model: str = "openai/gpt-4o",
        system_prompt: str | None = None,
    ):
        super().__init__(model=model)

        self.llm = ModuleLLM(
            api_key=api_key, llm_model=llm_model, system_prompt=system_prompt
        )

        self.memory = Memory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            api_key=api_key,
            llm_model=llm_model,
        )

        self.tool_manager = ToolManager(
            api_key=api_key,
            llm_model=llm_model,
            system_prompt=system_prompt,  # This will be changed
        )

        self.reasoning = reasoning(agent=self)

    def apply_plan(self, plan: Plan):
        tool_call_resp = self.tool_manager.call_tools(plan.llm_plan)
        self.memory.add_to_memory(
            type="Tool_Call_Responses", content=tool_call_resp, step=plan.step
        )
        return tool_call_resp
