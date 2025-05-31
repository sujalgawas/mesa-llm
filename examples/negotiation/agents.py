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
        chosen_brand=None,
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
        self.chosen_brand = chosen_brand

    def step(self):
        neighbor_cells = self.model.grid.get_neighborhood(
            pos=self.pos,
            moore=True,  # Set to False for only N/S/E/W neighbors
            include_center=False,
        )
        new_pos = self.random.choice(neighbor_cells)
        self.model.grid.move_agent(self, new_pos)
