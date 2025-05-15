from mesa_llm.module_llm import ModuleLLM


class LLMAgent:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        self.llm = ModuleLLM(api_key = api_key, model = model)






