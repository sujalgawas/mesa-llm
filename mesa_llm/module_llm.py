import os

from litellm import acompletion, completion, litellm
from tenacity import AsyncRetrying, retry, retry_if_exception_type, wait_exponential


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI, Gemini
    """

    def __init__(
        self,
        llm_model: str,
        api_key: str,
        api_base: str | None = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LLM module

        Args:
            api_key: The API key for the LLM provider
            llm_model: The model to use for the LLM in the format of {provider}/{LLM}
            system_prompt: The system prompt to use for the LLM
        """
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        provider = self.llm_model.split("/")[0].upper()
        os.environ[f"{provider}_API_KEY"] = self.api_key

        if not litellm.supports_function_calling(model=self.llm_model):
            print(
                f"Warning: {self.llm_model} does not support function calling. This model may not be able to use tools."
            )

    def get_messages(self, prompt: str | list[str]) -> list[dict]:
        """
        Format the prompt messages for the LLM of the form : {"role": ..., "content": ...}

        Args:
            prompt: The prompt to generate a response for

        Returns:
            The messages for the LLM
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if prompt:
            if isinstance(prompt, str):
                messages.append(
                    {"role": "user", "content": prompt},
                )
            elif isinstance(prompt, list):
                messages.extend([{"role": "user", "content": p} for p in prompt])

        return messages

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(
        self,
        prompt: str | list[str],
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        """
        Generate a response from the LLM using litellm based on the prompt

        Args:
            prompt: The prompt to generate a response for
            tool_schema: The schema of the tools to use
            tool_choice: The choice of tool to use
            response_format: The format of the response

        Returns:
            The response from the LLM
        """

        messages = self.get_messages(prompt)

        # If api_base is provided, use it to override the default API base
        if self.api_base:
            response = completion(
                model=self.llm_model,
                messages=messages,
                api_base=self.api_base,
                tools=tool_schema,
                tool_choice=tool_choice if tool_schema else None,
                response_format=response_format,
            )

        # Otherwise, use the default API base
        else:
            response = completion(
                model=self.llm_model,
                messages=messages,
                tools=tool_schema,
                tool_choice=tool_choice if tool_schema else None,
                response_format=response_format,
            )

        return response

    async def agenerate(
        self,
        prompt: str | list[str],
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        """
        Asynchronous version of generate() method for parallel LLM calls.
        """
        messages = self.get_messages(prompt)
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                response = await acompletion(
                    model=self.llm_model,
                    messages=messages,
                    tools=tool_schema,
                    tool_choice=tool_choice if tool_schema else None,
                    response_format=response_format,
                )
        return response
