import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class MemoryEntry:
    content: dict
    step: int

    def __str__(self) -> str:
        """
        Returns the memory entry as a string without formatting (simply an indented dict)
        """
        return str(json.dumps(self.content, indent=4))

    def style_format(self) -> str:
        """
        content is a dict that can have nested dictionaries of arbitrary depth
        """

        def format_nested_dict(data, indent_level=0):
            lines = []
            indent = "   " * indent_level

            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{indent}[blue]└──[/blue] [cyan]{key} :[/cyan]")
                    lines.extend(format_nested_dict(value, indent_level + 1))
                else:
                    lines.append(
                        f"{indent}[blue]└──[/blue] [cyan]{key} : [/cyan]{value}"
                    )

            return lines

        lines = []
        for key, value in [item for item in self.content.items() if item[1]]:
            lines.append(f"\n[bold cyan][{key.title()}][/bold cyan]")
            if isinstance(value, dict):
                lines.extend(format_nested_dict(value, 1))
            else:
                lines.append(f"   [blue]└──[/blue] [cyan]{value} :[/cyan]")

        content = "\n".join(lines)

        return content


class Memory:
    """
    Create a memory generic parent class that can be used to create different types of memories

    Attributes:
        agent : the agent that the memory belongs to
        api_key : the API key to use for the LLM
        llm_model : the model to use for the summarization
        display : whether to display the memory
    """

    def __init__(
        self,
        agent: "LLMAgent",
        api_key: str = os.getenv("OPENAI_API_KEY"),
        llm_model: str = "openai/gpt-4o-mini",
        display: bool = True,
    ):
        """
        Initialize the memory

        Args:
            api_key : the API key to use for the LLM
            llm_model : the model to use for the summarization
            agent : the agent that the memory belongs to
        """
        self.agent = agent
        self.llm = ModuleLLM(api_key=api_key, llm_model=llm_model)

        self.display = display

        self.system_prompt = """
        You are a helpful assistant that summarizes the short term memory into a long term memory.
        The long term memory should be a summary of the short term memory that is concise and informative.
        If the short term memory is empty, return the long term memory unchanged.
        If the long term memory is not empty, update it to include the new information from the short term memory.
        """

        self.llm.system_prompt = self.system_prompt

        self.step_content: dict = {}
        self.last_observation: dict = {}

    def add_to_memory(self, type: str, content: dict):
        """
        Add a new entry to the memory
        """
        if type == "observation":
            # Only store changed parts of observation
            changed_parts = {
                k: v for k, v in content.items() if v != self.last_observation.get(k)
            }
            if changed_parts:
                self.step_content[type] = changed_parts
            self.last_observation = content
        else:
            self.step_content[type] = content
