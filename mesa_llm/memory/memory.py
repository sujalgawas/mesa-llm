from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class MemoryEntry:
    content: dict
    step: int
    agent: "LLMAgent"

    def __str__(self) -> str:
        """
        Format the memory entry as a string.
        Note : 'content' is a dict that can have nested dictionaries of arbitrary depth
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

        return str(content)

    def display(self):
        if self.agent and hasattr(self.agent, "memory") and self.agent.memory.display:
            title = f"Step [bold purple]{self.agent.model.steps}[/bold purple] [bold]|[/bold] {type(self.agent).__name__} [bold purple]{self.agent.unique_id}[/bold purple]"
            panel = Panel(
                self.__str__(),
                title=title,
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
            )
            console = Console()
            console.print(panel)


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
        api_key: str | None = None,
        llm_model: str | None = None,
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
        if api_key and llm_model:
            self.llm = ModuleLLM(api_key=api_key, llm_model=llm_model)
        elif (not api_key and llm_model) or (api_key and not llm_model):
            raise ValueError("Both api_key and llm_model must be provided or neither")

        self.display = display

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
