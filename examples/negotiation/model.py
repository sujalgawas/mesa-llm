import math

from mesa.model import Model
from mesa.space import MultiGrid

from examples.negotiation.agents import BuyerAgent, SellerAgent
from mesa_llm.reasoning.reasoning import Reasoning


class NegotiationModel(Model):
    """
    A model for a negotiation game between a seller and a buyer.

    Args:
        initial_buyers (int): The number of initial buyers in the model.
        initial_sellers (int): The number of initial sellers in the model.
        width (int): The width of the grid.
        height (int): The height of the grid.
    """

    def __init__(
        self,
        initial_buyers: int,
        width: int,
        height: int,
        api_key: str,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height

        self.grid = MultiGrid(self.height, self.width, torus=False)

        # ---------------------Create the buyer agents---------------------
        buyer_system_prompt = "You are a buyer in a negotiation game. You are interested in buying a product from a seller. You are also interested in negotiating with the seller."
        buyer_internal_state = ""

        agents = BuyerAgent.create_agents(
            self,
            n=initial_buyers,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=buyer_system_prompt,
            vision=vision,
            internal_state=buyer_internal_state,
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_buyers,))
        y = self.rng.integers(0, self.grid.height, size=(initial_buyers,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        # ---------------------Create the seller agents---------------------
        seller_a = SellerAgent(
            model=self,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a Seller in a negotiation game. You are trying to pitch your product A to the Buyer type Agents. You are extremely good at persuading, and have good sales skills. You are also hardworking and dedicated to your work. To do any action, you must use the tools provided to you.",
            vision=vision,
            internal_state=["hardworking", "dedicated", "persuasive"],
        )
        self.grid.place_agent(
            seller_a,
            (math.floor(self.grid.width / 2), math.floor(self.grid.height / 2)),
        )

        seller_b = SellerAgent(
            model=self,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a Seller in a negotiation game. You are trying to pitch your product B to the Buyer type Agents. You are not interested in your work and are doing it for the sake of doing. To do any action, you must use the tools provided to you.",
            vision=vision,
            internal_state=["lazy", "unmotivated"],
        )
        self.grid.place_agent(
            seller_b,
            (math.floor(self.grid.width / 2), math.floor(self.grid.height / 2) + 1),
        )

    def step(self):
        """
        Execute one step of the model.
        """
        self.agents.shuffle_do("step")
