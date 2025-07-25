import os

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from examples.epstein_civil_violence.model import EpsteinModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

enable_automatic_parallel_stepping(mode="threading")

load_dotenv()

COP_COLOR = "#000000"

agent_colors = {
    CitizenState.ACTIVE: "#FE6100",
    CitizenState.QUIET: "#648FFF",
    CitizenState.ARRESTED: "#DB28A2",
}

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_citizens": 10,
    "initial_cops": 3,
    "width": 5,
    "height": 5,
    "api_key": os.getenv("GEMINI_API_KEY"),
    "reasoning": ReActReasoning,
    "llm_model": "gemini/gemini-1.5-flash",
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


def citizen_cop_portrayal(agent):
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


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


space_component = make_space_component(
    citizen_cop_portrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],  # Add ShowSalesButton here
        model_params=model_params,
        name="Espstein Civil Violence Model",
    )


"""run with:
conda activate mesa-llm && solara run examples/epstein_civil_violence/app.py
"""
