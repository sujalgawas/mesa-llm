from mesa_llm.module_llm import ModuleLLM


class LLMAgent:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o", system_prompt: str|None = None):
        """Initialize the LLMAgent with a ModuleLLM instance."""
        self.llm = ModuleLLM(api_key=api_key, model=model, system_prompt=system_prompt)

    def set_model(self, api_key: str, model: str = "openai/gpt-4o") -> None:
        """Set the model of the LLM."""
        self.llm = ModuleLLM(api_key=api_key, model=model, system_prompt=self.llm.system_prompt)
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for the LLM."""
        self.llm= ModuleLLM(api_key=self.llm.api_key, model=self.llm.model, system_prompt=system_prompt)