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

        self.system_prompt = system_prompt
        self.internal_state = internal_state

    def step(self):
        pass


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

        self.system_prompt = system_prompt
        self.internal_state = internal_state
