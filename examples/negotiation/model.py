import math

from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.model import Model

from examples.negotiation.agents import BuyerAgent, SellerAgent
from mesa_llm.reasoning import Reasoning


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

        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=False,
            capacity=math.inf,
            random=self.random,
        )

        # ---------------------Create the buyer agents---------------------
        buyer_system_prompt = "You are a buyer in a negotiation game."
        buyer_internal_state = ""

        BuyerAgent.create_agents(
            self,
            n=initial_buyers,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_buyers),
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=buyer_system_prompt,
            vision=vision,
            internal_state=buyer_internal_state,
        )

        # ---------------------Create the seller agents---------------------
        SellerAgent(
            model=self,
            cell=self.random.choice(self.grid.all_cells.cells),
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a Seller in a negotiation game. You are trying to pitch your product A to the Buyer type Agents. You are extremely good at persuading, and have good sales skills. You are also hardworking and dedicated to your work.",
            vision=vision,
            internal_state=["hardworking", "dedicated", "persuasive"],
        )

        SellerAgent(
            model=self,
            cell=self.random.choice(self.grid.all_cells.cells),
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a Seller in a negotiation game. You are trying to pitch your product B to the Buyer type Agents. You are not interested in your work and are doing it for the sake of doing.",
            vision=vision,
            internal_state=["lazy", "unmotivated"],
        )

    def step(self):
        """
        Execute one step of the model.
        """
        self.agents.shuffle_do("step")
