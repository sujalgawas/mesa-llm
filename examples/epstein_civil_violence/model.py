from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.epstein_civil_violence.agents import Citizen, Cop
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.integration_hooks import record_model


@record_model
class Model(Model):
    def __init__(
        self,
        initial_cops: int,
        initial_citizens: int,
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

        # ---------------------Create the cop agents---------------------
        cop_system_prompt = "You are a buyer in a negotiation game. You are interested in buying a product from a seller. You are also interested in negotiating with the seller. Prefer speaking over changing location as long as you have a seller in sight."

        agents = Cop.create_agents(
            self,
            n=initial_cops,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=cop_system_prompt,
            vision=vision,
            internal_state=None,
            budget=50,  # Each buyer has a budget of $50
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_cops,))
        y = self.rng.integers(0, self.grid.height, size=(initial_cops,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        # ---------------------Create the citizen agents---------------------
        agents = Citizen.create_agents(
            self,
            n=initial_citizens,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="",
            vision=vision,
            internal_state=None,
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_citizens,))
        y = self.rng.integers(0, self.grid.height, size=(initial_citizens,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

    def step(self):
        """
        Execute one step of the model.
        """

        print(
            f"\n[bold purple] step  {self.steps} ────────────────────────────────────────────────────────────────────────────────[/bold purple]"
        )
        self.agents.shuffle_do("step")
