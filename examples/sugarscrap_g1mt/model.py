import random

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.sugarscrap_g1mt.agents import Resource, Trader
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model


@record_model(output_dir="recordings")
class SugarScapeModel(Model):
    def __init__(
        self,
        initial_traders: int,
        initial_resources: int,
        width: int,
        height: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        parallel_stepping=True,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(self.width, self.height, torus=False)

        model_reporters = {
            "Trader_Count": lambda m: sum(1 for a in m.agents if isinstance(a, Trader)),
            "Total_Sugar": lambda m: sum(
                a.sugar for a in m.agents if isinstance(a, Trader)
            ),
            "Total_Spice": lambda m: sum(
                a.spice for a in m.agents if isinstance(a, Trader)
            ),
        }

        agent_reporters = {
            "sugar": lambda a: getattr(a, "sugar", None),
            "spice": lambda a: getattr(a, "spice", None),
            "mrs": lambda a: a.calculate_mrs() if isinstance(a, Trader) else None,
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        for _i in range(initial_resources):
            max_cap = random.randint(2, 5)
            resource = Resource(
                model=self, max_capacity=max_cap, current_amount=max_cap, growback=1
            )

            x = random.randrange(self.width)
            y = random.randrange(self.height)

            self.grid.place_agent(resource, (x, y))

        trader_system_prompt = (
            "You are a Trader agent in a Sugarscape simulation. "
            "You need Sugar and Spice to survive. "
            "If your MRS (Marginal Rate of Substitution) is high, you desperately need Sugar. "
            "If MRS is low, you need Spice. "
            "You can move to harvest resources or trade with neighbors."
        )

        agents = Trader.create_agents(
            self,
            n=initial_traders,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=trader_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Observe your inventory and MRS. Move to the best resource or propose a trade.",
        )

        x_pos = self.rng.integers(0, self.grid.width, size=(initial_traders,))
        y_pos = self.rng.integers(0, self.grid.height, size=(initial_traders,))

        for agent, i, j in zip(agents, x_pos, y_pos):
            agent.sugar = random.randint(5, 25)
            agent.spice = random.randint(5, 25)
            agent.metabolism_sugar = random.randint(1, 4)
            agent.metabolism_spice = random.randint(1, 4)

            agent.update_internal_metrics()

            self.grid.place_agent(agent, (i, j))

    def step(self):
        """
        Execute one step of the model.
        """
        print(
            f"\n[bold purple] step  {self.steps} ────────────────────────────────────────────────────────────────────────────────[/bold purple]"
        )

        self.agents.shuffle_do("step")

        self.datacollector.collect(self)


# ===============================================================
#                     RUN WITHOUT GRAPHICS
# ===============================================================

if __name__ == "__main__":
    """
    Run the model without the solara integration
    """
    from mesa_llm.reasoning.reasoning import Reasoning

    model = SugarScapeModel(
        initial_traders=5,
        initial_resources=20,
        width=10,
        height=10,
        reasoning=Reasoning(),
        llm_model="openai/gpt-4o-mini",
        vision=2,
    )

    for _ in range(5):
        model.step()
