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
        initial_sellers: int,
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
        # buyer_placement_cell=[(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(initial_buyers)]
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
        # seller_placement_cell = [(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(initial_sellers)]
        buyer_system_prompt = "You are a buyer in a negotiation game."
        seller_system_prompt = "You are a seller in a negotiation game."
        seller_internal_state = ""

        SellerAgent.create_agents(
            self,
            n=initial_sellers,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_sellers),
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=seller_system_prompt,
            vision=vision,
            internal_state=seller_internal_state,
        )

    def step(self):
        """
        Execute one step of the model.
        """
        self.agents.shuffle_do("step")
