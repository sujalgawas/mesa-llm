from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool


@tool
def set_chosen_brand(agent: LLMAgent, chosen_brand: str) -> str:
    """
    A tool to set the brand of choice of the buyer agent, It can either be brand A or brand B.

    Args:
        agent : The agent to set the brand of choice for.
        chosen_brand : The brand of choice of the buyer agent, either "A" or "B".

    Returns:
        str: The brand of choice of the buyer agent, either "A" or "B".
    """
    agent.chosen_brand = chosen_brand
    return f"Chosen brand of {agent} set to {chosen_brand}"


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

    def step(self):
        observation = self.generate_obs()
        prompt = "You are allowed to move around if you are not engaged in a conversation. Seller agents around you might try to pitch their product by sending you messages, take them into account and decide what to set yout chosen brand attribute as"
        plan = self.reasoning.plan(prompt=prompt, obs=observation)
        self.apply_plan(plan)
