# app.py
import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.sugarscrap_g1mt.agents import TraderState
from examples.sugarscrap_g1mt.model import SugarScapeModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

# Suppress Pydantic serialization warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)

# Also suppress through logging
logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")

load_dotenv()

COP_COLOR = "#000000"

agent_colors = {
    TraderState.Total_Count: "#FE6100",
    TraderState.Total_Sugar: "#648FFF",
    TraderState.Total_Spice: "#DB28A2",
}

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "inital_traders": 10,
    "initial_resources": 10,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/gemma 3:1b",
    "vision": 5,
    "parallel_stepping": True,
}

model = SugarScapeModel(
    initial_traders=model_params["inital_traders"],
    initial_resources=model_params["initial_resources"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
)


def trader_portrayal(agent):
    if agent is None:
        return
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Layer": 1,
        "Color": agent_colors.get(agent.state, "#FFFFFF"),
        "text": f"S:{agent.sugar} Sp:{agent.spice} MRS:{agent.calculate_mrs():.2f}",
        "text_color": "black",
    }

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


space_component = make_space_component(
    model,
    portrayal_function=trader_portrayal,
    grid_width=10,
    grid_height=10,
    post_process=post_process,
)

chart_component = make_plot_component(
    {state.name.lower(): state.name for state in TraderState}
)

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],
        model_params=model_params,
        name="SugarScape G1MT Example",
    )

    """
    run with
    cd examples/sugarscrap_g1mt
    conda activate mesa-llm && solara run app.py
    """
