"""
Demo script to compare sequential vs parallel stepping in random_bunnies example.

This script demonstrates the performance difference between traditional sequential
agent stepping and the new automatic parallel stepping feature.
"""

import os
import time

from dotenv import load_dotenv

from examples.random_bunnies.model import RandomBunniesModel
from mesa_llm.reasoning.react import ReActReasoning


def run_simulation(steps: int = 3, num_bunnies: int = 5):
    """
    Run a simulation with either sequential or parallel stepping.

    Args:
        steps: Number of simulation steps to run
        num_bunnies: Number of bunny agents to create

    Returns:
        Total time taken in seconds
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return None

    model = RandomBunniesModel(
        initial_bunnies=num_bunnies,
        width=10,
        height=10,
        api_key=api_key,
        reasoning=ReActReasoning,
        llm_model="openai/gpt-4o-mini",
        vision=2,
        seed=42,
        parallel_stepping=True,
    )

    start_time = time.time()

    for i in range(steps):
        step_start = time.time()
        print(f"\n--- Step {i + 1} ---")
        model.step()
        step_time = time.time() - step_start
        print(f"Step {i + 1} took {step_time:.2f} seconds")

    total_time = time.time() - start_time

    return total_time


def main():
    """
    Run parallel simulation.
    """

    num_bunnies = 50
    steps = 2

    parallel_time = run_simulation(steps=steps, num_bunnies=num_bunnies)

    if parallel_time is None:
        return

    print(f"Parallel time: {parallel_time:.2f} seconds")


if __name__ == "__main__":
    main()
