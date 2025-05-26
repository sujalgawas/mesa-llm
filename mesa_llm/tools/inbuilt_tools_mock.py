import inspect
import sys


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    return f"The current weather in {location} is {unit}."


def throw_stone(target: str, distance: float) -> str:
    """Make an agent throw a stone at a target"""
    return f"The agent threw a stone at {target} at a distance of {distance} meters."


# Get all the functions in the module that are not private into a list
inbuilt_tools = [
    obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    if not name.startswith("_")
]
