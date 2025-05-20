from mesa_llm.module_llm import ModuleLLM


class LLMAgent:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        """Initialize the LLMAgent with a ModuleLLM instance."""
        self.llm = ModuleLLM(api_key=api_key, model=model)

    def set_llm(self, api_key: str, model: str = "openai/gpt-4o") -> None:
        """Replace the current LLM with a new configuration."""
        self.llm = ModuleLLM(api_key=api_key, model=model)
