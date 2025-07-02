import os

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_space_component,
)

from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from examples.epstein_civil_violence.model import EpsteinModel
from mesa_llm.reasoning.react import ReActReasoning

load_dotenv()

COP_COLOR = "#000000"

agent_colors = {
    CitizenState.ACTIVE: "#FE6100",
    CitizenState.QUIET: "#648FFF",
    CitizenState.ARRESTED: "#808080",
}

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_citizens": 10,
    "initial_cops": 5,
    "width": 5,
    "height": 5,
    "api_key": os.getenv("OPENAI_API_KEY"),
    "reasoning": ReActReasoning,
    "llm_model": "openai/gpt-4o-mini",
    "vision": 5,
    "parallel_stepping": True,
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
    parallel_stepping=model_params["parallel_stepping"],
)

if __name__ == "__main__":

    def model_portrayal(agent):
        if agent is None:
            return

        portrayal = {
            "size": 50,
        }

        if isinstance(agent, Cop):
            portrayal["color"] = COP_COLOR

        elif isinstance(agent, Citizen):
            portrayal["color"] = agent_colors[agent.state]

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
