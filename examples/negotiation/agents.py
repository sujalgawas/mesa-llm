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
