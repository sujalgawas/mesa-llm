import os

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_space_component,
)

from examples.negotiation.agents import BuyerAgent, SellerAgent
from examples.negotiation.model import NegotiationModel
from mesa_llm.reasoning import ReActReasoning

load_dotenv()


def model_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 25,
    }

    if isinstance(agent, SellerAgent):
        portrayal["color"] = "tab:red"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    elif isinstance(agent, BuyerAgent):
        portrayal["color"] = "tab:blue"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 1

    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_buyers": 5,
    "width": 10,
    "height": 10,
    "api_key": os.getenv("GEMINI_API_KEY"),
    "reasoning": ReActReasoning,
    "llm_model": "gemini/gemini-2.0-flash",
    "vision": 1,
}

# simulator = ABMSimulator()  #am not too sure how this works, it wasn't working when I tried to run it, so I just coded a simple version, maybe you can include it?
# model = NegotiationModel(simulator=simulator)
model = NegotiationModel(
    initial_buyers=5,
    width=10,
    height=10,
    api_key=os.getenv("GEMINI_API_KEY"),
    reasoning=ReActReasoning,
    llm_model="gemini/gemini-2.0-flash",
    vision=1,
    seed=50,
)

page = SolaraViz(
    model,
    components=[make_space_component(model_portrayal)],
    model_params=model_params,
    name="Negotiation",
    # simulator=simulator,
)

page  # noqa
