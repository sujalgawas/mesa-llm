import os
from typing import TYPE_CHECKING, List, Optional

from mesa_llm.memory.memory import Memory, MemoryEntry
from collections import deque
if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

class ShortTermMemory(Memory):
    """
    Purely short-term memory class that remembers only the last n memory entries.

    Attributes:
        agent : the agent that the memory belongs to
        n : number of short-term memories to remember
        display : whether to display the memory
        api_key : the API key to use for the LLM
        llm_model : the model to use for the summarization
    """
    def __init__(
        self,
        agent: "LLMAgent",
        n: int = 5,
        display: bool = True,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        llm_model: str = "openai/gpt-4o-mini",
    ):
        super().__init__(
            agent=agent,
            api_key=api_key,
            llm_model=llm_model,
            display=display,
        )
        self.n = n
        self.short_term_memory = deque()
        self.system_prompt = """
            You are a helpful assistant that summarizes the short term memory into a long term memory.
            The long term memory should be a summary of the short term memory that is concise and informative.
            If the short term memory is empty, return the long term memory unchanged.
            If the long term memory is not empty, update it to include the new information from the short term memory.
            """

        if(self.agent.step_prompt):
            self.system_prompt+=" This is the prompt of the porblem you will be tackling:{self.agent.step_prompt}, ensure you summarize the short-term memory into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Add the new entry to the short term memory
        - Display the new entry
        """

        # Add the new entry to the short term memory
        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
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
                agent=self.agent,
                content=self.step_content,
                step=self.agent.model.steps,
            )

            self.short_term_memory.append(new_entry)
            self.step_content = {}

        # Display the new entry
        if self.display:
            new_entry.display()

    def format_short_term(self) -> str:
        """
        Get the short term memory
        """
        if not self.short_term_memory:
            return "No recent memory."

        else:
            lines = []
            for st_memory_entry in self.short_term_memory:
                lines.append(
                    f"Step {st_memory_entry.step}: \n{st_memory_entry.content}"
                )
            return "\n".join(lines)

    def __str__(self) -> str:
        return f"Short term memory:\n {self.format_short_term()}\n"
 

