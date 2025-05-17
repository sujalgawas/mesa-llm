import os

from litellm import completion


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM module

        Args:
            api_key: The API key for the LLM provider
            model: The model to use for the LLM
        """
        self.api_key = api_key
        provider = model.split("/")[0].upper()

        os.environ[f"{provider}_API_KEY"] = self.api_key

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a response from the LLM
        """
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = completion(model="openai/gpt-4o", messages=messages)
        return response


# test the module
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")  # Or simply your API key
    llm = ModuleLLM(api_key, "openai/gpt-4o")

    response = llm.generate("Hello, how are you?")
    print(response.choices[0].message.content)
