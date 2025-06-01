from mesa_llm.llm_agent import LLMAgent


class SellerAgent(LLMAgent):
    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
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

    def step(self):
        # observation = self.generate_observation()
        # prompt = "Look around you and go to grids where buyers are present, if there are any buyers in your cell or in the neighboring cells(at one cell distance), pitch them your product. Don't pitch to the same buyer agents again. "
        # plan = self.reasoning.plan(prompt=prompt, obs=observation)
        # self.apply_plan(plan)
        pass


class BuyerAgent(LLMAgent):
    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
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

    def step(self):
        observation = self.generate_observation()
        prompt = "Wait for a seller to pitch to you. If a seller pitches to you, respond to them with a message."
        plan = self.reasoning.plan(prompt=prompt, obs=observation)
        self.apply_plan(plan)
