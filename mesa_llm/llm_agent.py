from mesa_llm.module_llm import ModuleLLM


class LLMAgent:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o", system_prompt: str|None = None):
        """Initialize the LLMAgent with a ModuleLLM instance."""
        self.llm = ModuleLLM(api_key=api_key, model=model, system_prompt=system_prompt)

    def set_model(self, api_key: str, model: str = "openai/gpt-4o") -> None:
        """Set the model of the Agent."""
        self.llm.set_model(api_key=api_key, model=model)
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for the Agent."""
        self.llm.set_system_prompt(system_prompt=system_prompt)