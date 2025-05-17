import os

from litellm import completion


class ModuleLLM:
    """
    Currently supports OpenAI, Anthropic, xIA, Huggingface, Ollama, OpenRouter, NovitaAI
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        provider = model.split("/")[0].upper()

        os.environ[f"{provider}_API_KEY"] = self.api_key

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
        response = completion(
            model="openai/gpt-4o", 
            messages=messages
        )
        return response


# test the module
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")  # Or simply your API key
    llm = ModuleLLM(api_key, "openai/gpt-4o")

    response = llm.generate("Hello, how are you?")
    print(response.choices[0].message.content)
