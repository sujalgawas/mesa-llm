from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ReActReasoning(Reasoning):
    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def plan(self, prompt: str, obs: Observation, ttl: int = 1) -> Plan:
        """
        Plan the next (ReAct) action based on the current observation and the agent's memory.
        """
        step = obs.step + 1
        llm = self.agent.llm
        memory = self.agent.memory
        long_term_memory = memory.format_long_term()
        short_term_memory = memory.format_short_term()
        obs_str = str(
            obs
        )  # passing the latest obs separately from memory so that the llm can pay more attention to it.
        memory.add_to_memory(type="Observation", content=obs_str, step=step)

        system_prompt = f"""
        You are an autonomous agent in a simulation environment.
        You can think about your situation and take actions.
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
        {obs_str}

        ---

        # Instructions
        Based on your memory and current situation:
        1. Think through what is happening.
        2. Decide what you should do next.
        3. When you decide on an action, you use the available function calls to execute it. You must use the tools provided to you as a tool call.

        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt, tool_schema=self.agent.tool_manager.get_all_tools_schema()
        )
        response_message = rsp.choices[0].message
        react_plan = Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=str(react_plan), step=step)

        # --------------------------------------------------
        # Recording hook for plan event
        # --------------------------------------------------
        if self.agent.recorder is not None:
            self.agent.recorder.record_plan(
                agent_id=self.agent.unique_id,
                content={"plan": str(react_plan)},
            )

        return react_plan
