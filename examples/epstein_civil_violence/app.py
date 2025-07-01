import os

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_space_component,
)

from examples.epstein_civil_violence.agents import Citizen, Cop
from examples.epstein_civil_violence.model import EpsteinModel
from mesa_llm.reasoning.react import ReActReasoning

load_dotenv()


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_citizens": 20,
    "initial_cops": 5,
    "width": 10,
    "height": 10,
    "api_key": os.getenv("GEMINI_API_KEY"),
    "reasoning": ReActReasoning,
    "llm_model": "gemini/gemini-2.0-flash",
    "vision": 5,
}


model = EpsteinModel(
    initial_citizens=model_params["initial_citizens"],
    initial_cops=model_params["initial_cops"],
    width=model_params["width"],
    height=model_params["height"],
    api_key=model_params["api_key"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
)

if __name__ == "__main__":

    def model_portrayal(agent):
        if agent is None:
            return

        portrayal = {
            "size": 25,
        }

        if isinstance(agent, Cop):
            portrayal["color"] = "tab:red"
            portrayal["marker"] = "o"
            portrayal["zorder"] = 2
        elif isinstance(agent, Citizen):
            portrayal["color"] = "tab:blue"
            portrayal["marker"] = "o"
            portrayal["zorder"] = 1

        return portrayal

    page = SolaraViz(
        model,
        components=[
            make_space_component(model_portrayal),
        ],  # Add ShowSalesButton here
        model_params=model_params,
        name="Espstein Civil Violence Model",
    )


"""run with:
conda activate mesa-llm && solara run examples/epstein_civil_violence/app.py
"""
