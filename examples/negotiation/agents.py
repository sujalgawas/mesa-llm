from mesa.discrete_space import CellAgent

from mesa_llm.llm_agent import LLMAgent


class SellerAgent(LLMAgent, CellAgent):
    def __init__(
        self,
        model,
        cell,
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
        self.cell = cell

    def step(self):
        self.cell = self.cell.neighborhood.select_random_cell()
        observation = self.generate_observation()
        prompt = "Look around you and go to grids where buyers are present, if there are any buyers in your cell or in the neighboring cells(at one cell distance), pitch them your product. Don't pitch to the same buyer agents again. "
        plan = self.reasoning.plan(prompt=prompt, obs=observation)
        self.apply_plan(plan)


class BuyerAgent(LLMAgent, CellAgent):
    def __init__(
        self,
        model,
        cell,
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
        self.cell = cell
        self.chosen_brand = chosen_brand

    def step(self):
        self.cell = self.cell.neighborhood.select_random_cell()
