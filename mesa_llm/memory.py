import os
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from terminal_style import style

from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class MemoryEntry:
    type: str
    content: str
    step: int
    metadata: dict

    def __str__(self) -> str:
        return (
            style(
                f"[{self.type.title()} @ Step {self.step}] : ", color="green", bold=True
            )
            + self.content
        )


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
        self.short_term_memory = deque()
        self.long_term_memory = ""

        self.system_prompt = """
        You are a helpful assistant that summarizes the short term memory into a long term memory.
        The long term memory should be a summary of the short term memory that is concise and informative.
        If the short term memory is empty, return the long term memory unchanged.
        If the long term memory is not empty, update it to include the new information from the short term memory.
        """

        self.llm.set_system_prompt(self.system_prompt)

    def add_to_memory(
        self, type: str, content: str, step: int, metadata: dict | None = None
    ):
        """
        Add a new entry to the memory
        """
        metadata = metadata or {}
        new_entry = MemoryEntry(type, content, step, metadata)
        self.short_term_memory.append(new_entry)

        # Consolidate memory if the short term memory is over capacity
        if len(self.short_term_memory) > self.capacity + self.consolidation_capacity:
            memories_to_consolidate = [
                self.short_term_memory.popleft()
                for _ in range(self.consolidation_capacity)
            ]
            self.update_long_term_memory(memories_to_consolidate)
        agent_display_name = (
            self.agent.__class__.__name__ + " " + str(self.agent.unique_id) + " "
        )
        print(
            style("Added to the memory of ", color="green"),
            style(agent_display_name, color="cyan"),
            new_entry,
        )

    def format_short_term(self) -> str:
        if not self.short_term_memory:
            return "No recent memory."

        lines = [f"[{self.agent} Short-Term Memory]"]
        for entry in self.short_term_memory:
            lines.append(f"\n[{entry.type.title()} @ Step {entry.step}]")
            lines.append(entry.content.strip())
        return str("\n".join(lines))

    def format_long_term(self) -> str:
        """
        Get the long term memory
        """
        return str(self.long_term_memory)

    def update_long_term_memory(self, memories_to_consolidate: list[MemoryEntry]):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """
        entries = [self.convert_entry_to_dict(m) for m in memories_to_consolidate]

        prompt = f"""
            Short term memory:
                {entries}
            Long term memory:
                {self.long_term_memory}
            """

        self.long_term_memory = self.llm.generate(prompt)

    def convert_entry_to_dict(self, entry: MemoryEntry) -> dict:
        """
        Convert a memory entry to a dictionary
        """
        return {
            "type": entry.type,
            "content": entry.content,
            "step": entry.step,
            "metadata": entry.metadata,
        }
