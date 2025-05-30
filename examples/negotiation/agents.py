from mesa_llm.llm_agent import LLMAgent


class SellerAgent(LLMAgent):
    def __init__(
        self,
        space,
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
            space=space,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )

    def step(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        # Optionally, filter only empty cells
        empty_neighbors = [
            pos for pos in neighbors if self.model.grid.is_cell_empty(pos)
        ]
        if empty_neighbors:
            new_position = self.random.choice(empty_neighbors)
            self.model.grid.move_agent(self, new_position)


class BuyerAgent(LLMAgent):
    def __init__(
        self,
        space,
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
            space=space,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )

    def step(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        # Optionally, filter only empty cells
        empty_neighbors = [
            pos for pos in neighbors if self.model.grid.is_cell_empty(pos)
        ]
        if empty_neighbors:
            new_position = self.random.choice(empty_neighbors)
            self.model.grid.move_agent(self, new_position)
