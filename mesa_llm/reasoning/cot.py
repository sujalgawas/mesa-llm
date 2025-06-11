from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class CoTReasoning(Reasoning):
    """
    Use a chain of thought approach to decide the next action.
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def plan(self, prompt: str, obs: Observation, ttl: int = 1) -> Plan:
        """
        Plan the next (CoT) action based on the current observation and the agent's memory.
        """
        step = obs.step + 1
        llm = self.agent.llm
        memory = self.agent.memory
        long_term_memory = memory.format_long_term()
        short_term_memory = memory.format_short_term()
        obs_str = str(obs)

        # Add current observation to memory (for record)
        memory.add_to_memory(type="Observation", content=obs_str, step=step)

        system_prompt = f"""
        You are an autonomous agent operating in a simulation.
        Use a detailed step-by-step reasoning process (Chain-of-Thought) to decide your next action.
        Your memory contains information from past experiences, and your observation provides the current context.

        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {obs_str}

        ---

        # Instructions
        First think through the situation step-by-step, and explain it in the format given below.

        Thought 1: [Initial reasoning based on the observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]

        Keep the reasoning grounded in the current context and relevant history.
        **IMPORTANT**: When you decide on an action, use the available function calls to execute 1 action. You must use the tools provided to you as a tool call.
        **IMPORTANT**: There is no need to explicitly state the action in your reasoning in the message with your thoughts. The action will be inferred from the function call you make.
        ---
        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt, tool_schema=self.agent.tool_manager.get_all_tools_schema()
        )

        response_message = rsp.choices[0].message
        cot_plan = Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=str(cot_plan), step=step)

        return cot_plan
