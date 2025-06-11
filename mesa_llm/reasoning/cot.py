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
        Think in multiple reasoning steps before you act.
        **IMPORTANT**: When you decide on an action, you MUST use the available function calls to execute it. Do not just describe what you want to do - actually call the appropriate functions.

        Available functions include:
        - teleport_to_location: to move to a specific coordinate
        - speak_to: to send messages to other agents
        - set_chosen_brand: to set your brand preference (buyers only)

        Use the format below to respond:

        Thought 1: [Initial reasoning based on the observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [Use function calls to execute your chosen action - do not just describe it]

        Keep the reasoning grounded in the current context and relevant history.

        ---

        # Response:
        Thought 1:
        Thought 2:
        Thought 3:
        Thought 4:
        Action:
        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt, tool_schema=self.agent.tool_manager.get_all_tools_schema()
        )

        response_message = rsp.choices[0].message
        cot_plan = Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=str(cot_plan), step=step)

        return cot_plan
