import os

from litellm import completion, litellm


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI
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

    def generate(self, prompt: str, tool_schema: list[dict] | None = None) -> str:
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
                tool_choice="auto",
            )
        else:
            response = completion(model=self.llm_model, messages=messages)
        return response


# test the module
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("ready to go ------------------------------")

    api_key = os.getenv("GEMINI_API_KEY")  # Or simply your API key
    llm = ModuleLLM(api_key=api_key, llm_model="gemini/gemini-2.0-flash")

    response = llm.generate("Say 'hi'.")
    print(response.choices[0].message.content)
