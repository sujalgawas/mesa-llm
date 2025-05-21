from mesa_llm.memory import Memory
from mesa_llm.module_llm import ModuleLLM


class LLMAgent:
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

    Notes:
        - Each agent can only have one memory instance associated with it.
        - If no memory is passed at initialization, one can be attached later using `attach_memory()`.
        - Reassigning or replacing memory after it's been attached is not allowed and will raise a ValueError.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o",
        system_prompt: str | None = None,
        memory: Memory | None = None,
    ):
        self.llm = ModuleLLM(api_key=api_key, model=model, system_prompt=system_prompt)
        self._memory = Memory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            api_key=api_key,
            llm_model=model,
        )

    def set_model(self, api_key: str, model: str = "openai/gpt-4o") -> None:
        """Set the model of the Agent."""
        self.llm.set_model(api_key=api_key, model=model)

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for the Agent."""
        self.llm.set_system_prompt(system_prompt=system_prompt)
