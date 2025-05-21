import os

from litellm import completion


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI
    """

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None):
        """
        Initialize the LLM module

        Args:
            api_key: The API key for the LLM provider
            model: The model to use for the LLM in the format of {provider}/{model}
            system_prompt: The system prompt to use for the LLM
        """
        self.api_key = api_key
        self.model=model
        self.system_prompt = system_prompt
        provider = self.model.split("/")[0].upper()
        os.environ[f"{provider}_API_KEY"] = self.api_key

    def set_system_prompt(self, system_prompt: str):
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
    
    def set_model(self, api_key: str, model: str):
        """Set or update the model and API key."""
        self.api_key = api_key
        self.model = model
        provider = self.model.split("/")[0].upper()
        os.environ[f"{provider}_API_KEY"] = self.api_key

    def generate(self, prompt: str) -> str:
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        response = completion(model=self.model, messages=messages)
        return response


# test the module
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")  # Or simply your API key
    llm = ModuleLLM(api_key, "openai/gpt-4o")

    response = llm.generate("Hello, how are you?")
    print(response.choices[0].message.content)
