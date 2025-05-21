from mesa_llm.module_llm import ModuleLLM
from mesa_llm.memory import Memory #to be done

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

    Methods:
        attach_memory(memory):
            Attach a memory instance to this agent. Raises an error if a memory is already attached.

        set_model(api_key, model):
            Update the LLM model used by the agent.

        set_system_prompt(system_prompt):
            Update the system prompt used in completions.

    Notes:
        - Each agent can only have one memory instance associated with it.
        - If no memory is passed at initialization, one can be attached later using `attach_memory()`.
        - Reassigning or replacing memory after it's been attached is not allowed and will raise a ValueError.
    """
    

    def __init__(self, api_key: str, model: str = "openai/gpt-4o", system_prompt: str|None = None, memory: Memory|None =None):
        self.llm = ModuleLLM(api_key=api_key, model=model, system_prompt=system_prompt)
        self._memory = None
        if memory is not None:
            self.attach_memory(memory)

    @property
    def memory(self):
        return self._memory

    def attach_memory(self, memory):
        if self._memory is not None:
            raise ValueError("Memory already attached to this agent.")
        self._memory = memory

    def set_model(self, api_key: str, model: str = "openai/gpt-4o") -> None:
        """Set the model of the Agent."""
        self.llm.set_model(api_key=api_key, model=model)
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for the Agent."""
        self.llm.set_system_prompt(system_prompt=system_prompt)