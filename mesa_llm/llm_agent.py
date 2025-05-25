from mesa.agent import Agent
from mesa.model import Model

from mesa_llm.memory import Memory
from mesa_llm.module_llm import ModuleLLM


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        api_key (str): The API key for the LLM provider.
        model (str): The model to use for the LLM in the format 'provider/model'. Defaults to 'openai/gpt-4o'.
        system_prompt (str | None): Optional system prompt to be used in LLM completions.
        memory (Memory | None): Optional memory instance to attach to this agent. Can only be set once.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.

    """

    def __init__(
        self,
        model: Model,
        api_key: str,
        llm_model: str = "openai/gpt-4o",
        system_prompt: str | None = None,
    ):
        super().__init__(model=model)

        self.llm = ModuleLLM(
            api_key=api_key, llm_model=llm_model, system_prompt=system_prompt
        )
        self._memory = Memory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            api_key=api_key,
            llm_model=llm_model,
        )
