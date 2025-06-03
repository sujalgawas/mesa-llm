from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

seller_tool_manager = ToolManager()
buyer_tool_manager = ToolManager()


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

        self.tool_manager = seller_tool_manager

    def step(self):
        observation = self.generate_obs()
        prompt = "Look around you and go to grids where buyers are present, if there are any buyers in your cell or in the neighboring cells, pitch them your product. Don't pitch to the same buyer agents again. "
        plan = self.reasoning.plan(prompt=prompt, obs=observation)
        self.apply_plan(plan)


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
        self.chosen_brand = None
        self.tool_manager = buyer_tool_manager

    def step(self):
        observation = self.generate_obs()
        prompt = "Move around if you are not engaged in a conversation. Seller agents around you might try to pitch their product by sending you messages, take them into account and decide what to set yout chosen brand attribute as"
        plan = self.reasoning.plan(prompt=prompt, obs=observation)
        self.apply_plan(plan)
