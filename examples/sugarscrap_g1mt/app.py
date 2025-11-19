# app.py

from dotenv import load_dotenv

from examples.sugarscrap_g1mt.agents import Trader
from mesa_llm.reasoning.react import ReActReasoning

load_dotenv()

COP_COLOR = "#000000"

agent_colors = {
    Trader.ACTIVE: "#FE6100",
    Trader.QUIET: "#648FFF",
    Trader.ARRESTED: "#DB28A2",
}

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
    "reasoning": ReActReasoning,
    "llm_model": "openai/gpt-4o-mini",
    "vision": 5,
    "parallel_stepping": True,
}
