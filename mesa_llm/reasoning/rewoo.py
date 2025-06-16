from typing import TYPE_CHECKING

# import json
# from pydantic import BaseModel, Field, model_validator
from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


# @dataclass
# class ReWOOOutput(BaseModel):
#     plan: str
#     step_1: str = Field(description="First action with expected outcome")
#     step_2: Optional[str] = Field(None, description="Second action building on Step 1")
#     step_3: Optional[str] = Field(None, description="Third action if needed")
#     step_4: Optional[str] = Field(None, description="Fourth action if needed")
#     step_5: Optional[str] = Field(None, description="Final action if needed")
#     contingency: str

#     @model_validator(mode='after')
#     def validate_consecutive_steps(self):
#         # Get all step values
#         steps = [self.step_1, self.step_2, self.step_3, self.step_4, self.step_5]

#         # Find the last non-None step
#         last_step_index = -1
#         for i, step in enumerate(steps):
#             if step is not None:
#                 last_step_index = i

#         # Ensure no gaps in steps (no None values before the last step)
#         for i in range(last_step_index):
#             if steps[i] is None:
#                 raise ValueError(f"Steps must be consecutive. Found None at step_{i+1} but step_{last_step_index+1} has a value.")

#         return self


class ReWOOReasoning(Reasoning):
    """
    ReWOO is a reasoning approach that creates a plan that can be executed without needing new observations.
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def plan(self, prompt: str, obs: Observation, ttl: int = 1) -> Plan:
        """
        Plan the next (ReWOO) action based on the current observation and the agent's memory.
        """
        llm = self.agent.llm
        memory = self.agent.memory
        long_term_memory = memory.format_long_term()
        short_term_memory = memory.format_short_term()

        system_prompt = f"""
        You are an autonomous agent that creates multi-step plans without re-observing during execution.
        Using the ReWOO (Reasoning WithOut Observation) approach, you will create a comprehensive plan
        that anticipates multiple steps ahead based on your current observation and memory.

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
        Create a detailed multi-step plan that can be executed without needing new observations.
        Your plan should anticipate likely scenarios and include contingencies.

        Determine the optimal number of steps (1-5) based on the complexity of the task and available tools.
        Use this format:


            "plan": "Describe your overall strategy and reasoning",
            "step_1": "First action with expected outcome",
            "step_2": "Second action building on Step 1 (optional)",
            "step_3": "Third action if needed (optional)",
            "step_4": "Fourth action if needed (optional)",
            "step_5": "Final action if needed (optional)",
            "contingency": "What to do if things don't go as expected"


        Only include the steps you need (step_1 is required, step_2 through step_5 are optional).
        Set unused step fields to null. The plan should be comprehensive enough to execute
        for multiple simulation steps without requiring new environmental observations.
        Refer to available tools when planning actions.

        ---
        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(),
            tool_choice="none",
            # response_format=ReWOOOutput
        )

        memory.add_to_memory(type="Plan", content=rsp.choices[0].message.content)

        rewoo_plan = self.execute_tool_call(rsp.choices[0].message.content)

        # --------------------------------------------------
        # Recording hook for plan event
        # --------------------------------------------------
        if self.agent.recorder is not None:
            self.agent.recorder.record_event(
                event_type="plan",
                content={"plan": str(rewoo_plan)},
                agent_id=self.agent.unique_id,
            )

        return rewoo_plan
