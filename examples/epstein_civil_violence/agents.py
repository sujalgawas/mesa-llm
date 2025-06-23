import random

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

citizen_tool_manager = ToolManager()
cop_tool_manager = ToolManager()


class Citizen(LLMAgent):
    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state=None,
        regime_legitimacy=0.5,
    ):
        # Define state mapping
        state = {1: "quiet", 2: "active", 3: "arrested"}

        # Risk aversion score (random float between 0 and 1)
        self.risk_aversion = random.random()

        random_value = random.randint(1, 3)

        # Set internal state if not provided
        self.internal_state = internal_state or [
            state[random_value],
            f"The tendency of the agent to be averse to risk is {self.risk_aversion:.2f} out of 1",
        ]

        # Call the superclass constructor with updated internal state
        super().__init__(
            model=model,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=self.internal_state,
        )

        self.regime_legitimacy = regime_legitimacy
        self.jail_sentence = 0

    def step(self):
        observation = self.generate_obs()
        prompt = ""
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["move_one_step", "join the rebels"],
        )
        self.apply_plan(plan)


class Cop(LLMAgent):
    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        max_jail_term=2,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.max_jail_term = max_jail_term

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        observation = self.generate_obs()
        prompt = "Inspect your local vision and arrest a random active agent. Move if applicable."
        plan = self.reasoning.plan(
            prompt=prompt, obs=observation, selected_tools=["arrest", "move_one_step"]
        )
        self.apply_plan(plan)
