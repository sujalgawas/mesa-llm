import os

from dotenv import load_dotenv
from mesa import Model
from mesa.space import MultiGrid

from examples.random_bunnies.agents import Bunny
from mesa_llm.reasoning.react import ReActReasoning


class RandomBunniesModel(Model):
    def __init__(
        self,
        width,
        height,
        initial_bunnies,
        api_key,
        reasoning,
        llm_model,
        vision,
        seed,
        parallel_stepping=False,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.parallel_stepping = parallel_stepping

        agents = Bunny.create_agents(
            self,
            n=initial_bunnies,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            vision=vision,
            internal_state="",
        )

        x = self.rng.integers(0, self.width, size=(initial_bunnies,))
        y = self.rng.integers(0, self.height, size=(initial_bunnies,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

    def step(self):
        """
        Execute one step of the model.

        The parallel_stepping flag automatically controls whether agents
        step sequentially or in parallel - no code changes needed!
        """
        self.agents.shuffle_do("step")


# ===============================================================
#                     RUN WITHOUT GRAPHICS
# ===============================================================


if __name__ == "__main__":
    """
    run with
    conda activate mesa-llm
    python -m examples.random_bunnies.model

    To test parallel stepping, change parallel_stepping to True in model_params
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    llm_model = "openai/gpt-4o-mini"

    model_params = {
        "seed": {
            "type": "InputText",
            "value": 42,
            "label": "Random Seed",
        },
        "initial_bunnies": 3,
        "width": 10,
        "height": 10,
        "api_key": api_key,
        "reasoning": ReActReasoning,
        "llm_model": llm_model,
        "vision": 2,
        "parallel_stepping": False,  # Set to True to enable parallel stepping
    }

    model = RandomBunniesModel(
        initial_bunnies=model_params["initial_bunnies"],
        width=model_params["width"],
        height=model_params["height"],
        api_key=model_params["api_key"],
        reasoning=model_params["reasoning"],
        llm_model=model_params["llm_model"],
        vision=model_params["vision"],
        seed=model_params["seed"]["value"],
        parallel_stepping=model_params["parallel_stepping"],
    )

    for i in range(3):
        print(f"\n--- Model Step {i + 1} ---")
        model.step()
