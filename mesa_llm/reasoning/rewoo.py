from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


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
        step = obs.step + 1
        llm = self.agent.llm
        memory = self.agent.memory
        long_term_memory = memory.format_long_term()
        short_term_memory = memory.format_short_term()
        obs_str = str(obs)

        # Add current observation to memory
        memory.add_to_memory(type="Observation", content=obs_str, step=step)

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
        {obs_str}

        ---

        # Instructions
        Create a detailed multi-step plan that can be executed without needing new observations.
        Your plan should anticipate likely scenarios and include contingencies.

        Use this format:

        Plan: [Describe your overall strategy and reasoning]

        Step 1: [First action with expected outcome]
        Step 2: [Second action building on Step 1]
        Step 3: [Third action if needed]
        Step 4: [Fourth action if needed]
        Step 5: [Final action if needed]

        Contingency: [What to do if things don't go as expected]

        The plan should be comprehensive enough to execute for multiple simulation steps
        without requiring new environmental observations.
        Refer to available tools when planning actions.

        ---

        # Response:
        Plan:
        Step 1:
        Step 2:
        Step 3:
        Step 4:
        Step 5:
        Contingency:
        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt, tool_schema=self.agent.tool_manager.get_all_tools_schema()
        )

        response_message = rsp.choices[0].message

        rewoo_plan = Plan(step=step, llm_plan=response_message, ttl=ttl)
        memory.add_to_memory(type="Plan", content=str(rewoo_plan), step=step)

        if self.agent.recorder is not None:
            self.agent.recorder.record_event(
                event_type="plan",
                content={"plan": str(rewoo_plan)},
                agent_id=self.agent.unique_id,
            )

        return rewoo_plan
