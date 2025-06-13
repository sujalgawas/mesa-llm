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
        {obs_str}

        ---

        # Instructions
        Based on the information above, think about what you should do with proper reasoning, And then decide your plan of action. Respond in the
        following format:
        Reasoning: [Your reasoning about the situation, including how your memory informs your decision]
        Action: [The action you decide to take]

        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(),
            tool_choice="none",
        )

        chaining_message = rsp.choices[0].message.content
        memory.add_to_memory(type="Plan", content=chaining_message, step=step)
        system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(),
        )
        response_message = rsp.choices[0].message
        react_plan = Plan(step=step, llm_plan=response_message, ttl=1)

        memory.add_to_memory(type="Plan-Execution", content=str(react_plan), step=step)

        return react_plan
