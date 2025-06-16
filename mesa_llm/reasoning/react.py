import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class ReActOutput(BaseModel):
    reasoning: str
    action: str


class ReActReasoning(Reasoning):
    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def plan(self, prompt: str, obs: Observation, ttl: int = 1) -> Plan:
        """
        Plan the next (ReAct) action based on the current observation and the agent's memory.
        """
        llm = self.agent.llm
        memory = self.agent.memory
        long_term_memory = memory.format_long_term()
        short_term_memory = memory.format_short_term()
        obs_str = str(
            obs
        )  # passing the latest obs separately from memory so that the llm can pay more attention to it.
        memory.add_to_memory(type="Observation", content=obs_str)

        system_prompt = f"""
        You are an autonomous agent in a simulation environment.
        You can think about your situation and describe your plan.
        Use your short-term and long-term memory to guide your behavior.
        You should also use the current observation you have made of the environrment to take suitable actions.
        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {obs}

        ---

        # Instructions
        Based on the information above, think about what you should do with proper reasoning, And then decide your plan of action. Respond in the
        following format:
        reasoning: [Your reasoning about the situation, including how your memory informs your decision]
        action: [The action you decide to take - Do NOT use any tools here, just describe the action you will take]

        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(),
            tool_choice="none",
            response_format=ReActOutput,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)

        memory.add_to_memory(type="Plan", content=formatted_response)

        react_plan = self.execute_tool_call(formatted_response["action"])

        # --------------------------------------------------
        # Recording hook for plan event
        # --------------------------------------------------
        if self.agent.recorder is not None:
            self.agent.recorder.record_event(
                event_type="plan",
                content={"plan": str(react_plan)},
                agent_id=self.agent.unique_id,
            )

        return react_plan
