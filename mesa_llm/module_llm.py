import os

from litellm import completion, litellm


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI, Gemini
    """

    def __init__(self, api_key: str, llm_model: str, system_prompt: str | None = None):
        """
        Initialize the LLM module

        Args:
            api_key: The API key for the LLM provider
            llm_model: The model to use for the LLM in the format of {provider}/{LLM}
            system_prompt: The system prompt to use for the LLM
        """
        self.api_key = api_key
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        provider = self.llm_model.split("/")[0].upper()
        os.environ[f"{provider}_API_KEY"] = self.api_key

        if not litellm.supports_function_calling(model=self.llm_model):
            print(
                f"Warning: {self.llm_model} does not support function calling. This model may not be able to use tools."
            )

    def set_system_prompt(self, system_prompt: str):
        """Set or update the system prompt."""
        self.system_prompt = system_prompt

    def generate(
        self,
        prompt: str,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        if tool_schema:
            response = completion(
                model=self.llm_model,
                messages=messages,
                tools=tool_schema,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        else:
            response = completion(
                model=self.llm_model, messages=messages, response_format=response_format
            )
        return response
