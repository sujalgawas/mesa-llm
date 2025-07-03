import json
import os
from collections import deque
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
    Create a memory object that stores the agent's short and long term memory

    Attributes:
        agent : the agent that the memory belongs to

    Memory is composed of
        - A short term memory who stores the n (int) most recent interactions (observations, planning, discussions)
        - A long term memory that is a summary of the memories that are removed from short term memory (summary
        completed/refactored as it goes)

    """

    def __init__(
        self,
        agent: "LLMAgent",
        short_term_capacity: int = 5,
        consolidation_capacity: int = 2,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        llm_model: str = "openai/gpt-4o-mini",
        display: bool = True,
    ):
        """
        Initialize the memory

        Args:
            short_term_capacity : the number of interactions to store in the short term memory
            api_key : the API key to use for the LLM
            llm_model : the model to use for the summarization
            agent : the agent that the memory belongs to
        """
        self.agent = agent
        self.llm = ModuleLLM(api_key=api_key, llm_model=llm_model)

        self.capacity = short_term_capacity
        self.consolidation_capacity = consolidation_capacity
        self.display = display

        self.short_term_memory = deque()
        self.long_term_memory = ""

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

    def _update_long_term_memory(self):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """

        prompt = f"""
            Short term memory:
                {self.format_short_term()}
            Long term memory:
                {self.long_term_memory}
            """

        self.long_term_memory = self.llm.generate(prompt)

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Add the new entry to the short term memory
        - Consolidate the memory if the short term memory is over capacity
        - Display the new entry
        """

        # Add the new entry to the short term memory
        if pre_step:
            new_entry = MemoryEntry(
                content=self.step_content,
                step=None,
            )
            self.short_term_memory.append(new_entry)
            self.step_content = {}
            return

        elif not self.short_term_memory[-1].content.get("step", None):
            pre_step = self.short_term_memory.pop()
            self.step_content.update(pre_step.content)
            new_entry = MemoryEntry(
                content=self.step_content,
                step=self.agent.model.steps,
            )

            self.short_term_memory.append(new_entry)
            self.step_content = {}

        # Consolidate memory if the short term memory is over capacity
        if len(self.short_term_memory) > self.capacity + self.consolidation_capacity:
            self.short_term_memory.popleft()
            self._update_long_term_memory()

        if self.display:
            # Display the new entry
            title = f"Step [bold purple]{self.agent.model.steps}[/bold purple] [bold]|[/bold] {type(self.agent).__name__} [bold purple]{self.agent.unique_id}[/bold purple]"
            panel = Panel(
                new_entry.style_format(),
                title=title,
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
            )
            console = Console()
            console.print(panel)

    def format_short_term(self) -> str:
        if not self.short_term_memory:
            return "No recent memory."

        lines = [f"[{self.agent} Short-Term Memory]"]
        for entry in self.short_term_memory:
            lines.append(f"\n[Step {entry.step}]")
            lines.append(str(entry.content))
        return str("\n".join(lines))

    def format_long_term(self) -> str:
        """
        Get the long term memory
        """
        return str(self.long_term_memory)

    def __str__(self) -> str:
        return f"Short term memory:\n {self.format_short_term()}\n\nLong term memory: \n{self.format_long_term()}"
