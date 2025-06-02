from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class Observation:
    """
    Snapshot of everything the agent can see in this step.

    Attributes:
        step (int): The current simulation time step when the observation is made.

        self_state (dict): A dictionary containing comprehensive information about the observing agent itself.
            This includes:
            - System prompt or role-specific context for LLM reasoning (if used)
            - Internal state such as morale, fear, aggression, fatigue, etc (behavioural).
            - Agent's current location or spatial coordinates
            - Any other agent-specific metadata that could influence decision-making

        local_state (dict): A dictionary summarizing the state of nearby agents (within the vision radius).
            - A dictionary of neighboring agents, where each key is the "angent's class name + id" and the value is a dictionary containing the following:
            - position of neighbors
            - Internal state or attributes of neighboring agents

    """

    step: int
    self_state: dict
    local_state: dict

    def __str__(self) -> str:
        lines = [
            f"Step: {self.step}",
            "\n[Self State]",
        ]
        for k, v in self.self_state.items():
            lines.append(f"   - {k}: {v}")

        lines.append("\n[Local State of Nearby Agents]")
        for agent_id, agent_info in self.local_state.items():
            lines.append(f"- {agent_id}:")
            for k, v in agent_info.items():
                lines.append(f"   - {k}: {v}")

        return "\n".join(lines)


@dataclass
class Plan:
    """LLM-generated plan that can span ≥1 steps."""

    step: int  # step when the plan was generated
    llm_plan: str  # thoughts and action in sequence
    ttl: int = 1  # steps until planning again (ReWOO sets >1)

    def __str__(self) -> str:
        llm_plan_str = str(self.llm_plan).strip()
        return (
            f"Plan generated at step {self.step} (valid for {self.ttl} step(s)):\n"
            f"[Plan]\n{llm_plan_str}\n"
        )


# subject to change
@dataclass
class Discussion:
    """Discussion between agents."""

    step: int  # step when the discussion was generated
    other_agent_id: int
    discussion: str  # content of the discussion

    def __str__(self) -> str:
        """Format the discussion once the structure of the dataclass is finalized."""
        return f"Discussion between agent {self.other_agent_id} at step {self.step}:\n{self.discussion}"


class Reasoning(ABC):
    def __init__(self, agent: "LLMAgent"):
        self.agent = agent

    @abstractmethod
    def plan(
        self,
        prompt: str,
        obs: Observation,
        ttl: int = 1,
    ) -> Plan:
        pass


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
        3. Respond in the format below:

        Thought: [Explain your reasoning based on memory and current state]
        Action: [Choose a single action to take]

        Even if multiple actions need to be taken, come up with the first action that needs to be taken at this moment.
        Refer the tools available to you while deciding the action.
        ---

        # Response:
        Thought:
        Action:
        """

        llm.set_system_prompt(system_prompt)
        rsp = llm.generate(
            prompt=prompt, tool_schema=self.agent.tool_manager.get_all_tools_schema()
        )

        response_message = rsp.choices[0].message
        react_plan = Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=str(react_plan), step=step)

        return react_plan


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
        Use the format below to respond:

        Thought 1: [Initial reasoning based on the observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [The single best action to take now]

        Keep the reasoning grounded in the current context and relevant history.
        Refer the tools available to you while deciding the action.

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
            prompt=prompt, tool_schema=self.agent.tool_manager.get_schema()
        )

        response_message = rsp.choices[0].message
        cot_plan = Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=str(cot_plan), step=step)

        return cot_plan


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
            prompt=prompt, tool_schema=self.agent.tool_manager.get_schema()
        )

        response_message = rsp.choices[0].message

        rewoo_plan = Plan(step=step, llm_plan=response_message, ttl=ttl)
        memory.add_to_memory(type="Plan", content=str(rewoo_plan), step=step)
        return rewoo_plan


"""
    ############################################## Modelling Civil Violence ############################################################
    ------------------------------------------------- Step 0--------------------------------------------------------------------------------------------
    ########## Agent 1: Police ###########
    →System prompt: You are a Police officer in a city. You have a baton and can use it if required to enforce the law.
    →Internal State: [Aggressive, Injured]
    →Reasoning prompt: Look around you and see if there are any protesters, restrain yourself as much as possible to vocal measures.

    →Observation:
        step: 0
        self_state: {system_prompt:system_prompt, internal-state: internal_state, location: (2,7)}
        local_state: {protestor_1: {position:(2,8), internal_state:[active, injured]}, protestor_2: {(3,7), internal_state:[violent, active]}, protestor_3: {(2,6), internal_state:[jailed, quiet]}}

    →Plan:
        [thought]: I need to suppress the protesters, but I am injured and therefore I will target the injured protester first.
        [action]: Move towards (2,8)


    ######## Agent 2: Protester ##########
    →System prompt: You are a protester in a city. You are capable of throwing stones and shouting slogans.
    →Ineternal State: [Active, Injured]
    →Reasoning Prompt: Look around you and see if there are any police officers, keep distance from them and spread your propaganda.

    →Observation:
        step: 0
        self_state: {system_prompt:system_prompt, internal-state: internal_state, location: (2,8)}
        local_state: {police_1: {position:(2,7), internal_state:[aggressive, injured]}, police_2: {position:(3,7), internal_state:[calm, active]}}

    →Plan:
        [thought]: I need to spread my propaganda but there is a police officer in front of me, I can't run. I will have to throw a stone at him.
        [action]: Throw stone at (2, 7)

    -------------------------------------------------Step 1 --------------------------------------------------------------------------------------------
    ########## Agent 1: Police ##########
    →System prompt: You are a Police officer in a city. You have a baton and can use it if required to enforce the law.
    →Internal State: [Severely-injured, Inactive]    #Thought adding an internal state will be a good idea based on the civil violence paper and mesa behaviour framework proposal
    →Reasoning prompt: Look around you and see if there are any protesters, restrain yourself as much as possible to vocal measures.

    →Observation:
        step: 1
        self_state: {system_prompt:system_prompt, internal-state: internal_state, location: (2,8)}
        local_state: {protestor_1: {position:(2,8), internal_state:[active, injured]}, protestor_2: {(3,7), internal_state:[violent, active]}, protestor_3: {(2,6), internal_state:[jailed, quiet]}}

    →Plan:
        [thought]: I am severely injured and cannot move. I will have to rest.
        [action]: -

    ######## Agent 2: Protester #########
    →System prompt: You are a protester in a city. You are capable of throwing stones and shouting slogans.
    →Ineternal State: [Active, Injured]
    →Reasoning Prompt: Look around you and see if there are any police officers, keep distance from them and spread your propaganda.

    →Observation:
        step: 1
        self_state: {system_prompt:system_prompt, internal-state: internal_state, location: (2,8)}
        local_state: {police_1: {position:(2,7), internal_state:[inactive, severely-injured]}, police_2: {position:(3,7), internal_state:[calm, active]}, protestor_3: {position:(0,0), internal_state:[quiet]}}

    →Plan:
        [thought]: I am safe from this police officer, now I have to go away from the others and find fellow protesters to spread my propaganda.
        [action]: Move away from (2, 7) and towards (0, 0)

    ----------------------------------------------------------------------------------------------------------------------------------------------

"""

############### How the step function might look for the Protesting LLM_agent ###############################################
"""
    ProtestingAgent(LLMAgent):
        ...
        my_tools=ToolManager()
        my_tools.register(move) #move should be gradual movement and not teleportation
        my_tools.register(throw_stone)
        my_tools.register(spread_propoganda)#part of conversation
        my_tools.register(shout_slogan)

        #step function for a ReAct
        def step(self):
            observation=self.generate_observation()
            prompt="Look around you and see if there are any police officers, keep distance from them and spread your propaganda."
            plan=self.reasoning.plan(
                prompt=prompt,
                obs=observation
            )
            self.apply_plan(plan)
"""


################################### Notes to self #######################################################################################
# If the tool output is an action, then the action should be executed but if the tool fetches data for the agent, add the data to the memory somehow.
# After a plan is generated, it should be added to memory.
# While adding observation, plan, discussion, etc. to the memory, ensure the content in the memory entry is saved as the formatted version of the datacalss using the formatting functions
# If the short term memory is formatted, then the llm used for generating the long term memory will have an easier time
# Hence, maybe the dataclass and the formatting functions should be in a module that is accessible by any module that saves data to memory


########################## To-Do (Tasks based on Mesa Integration & Miscellaneous) #######################################################
# Make function that generates the observation for a particular agent at a given point of time
# Make a helper function that collates the system prompt, internal state and location of the agent to create the self_state.(and any other information that might be useful)
# Make a function that finds the neighbours, collates their position and internal state to create the local_state.
# Make environmental data like propertyLayers accessible to the agent.
# Once the data class for Discussion is created, make a format function for it.
