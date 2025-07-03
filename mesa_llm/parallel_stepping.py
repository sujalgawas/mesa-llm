"""
Automatic parallel stepping for Mesa-LLM simulations.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from mesa.agent import Agent, AgentSet

if TYPE_CHECKING:
    from .llm_agent import LLMAgent

# Global variable to control parallel stepping mode
_PARALLEL_STEPPING_MODE = "asyncio"  # or "threading"


async def step_agents_parallel(agents: list[Agent | LLMAgent]) -> None:
    """Step all agents in parallel using async/await."""
    tasks = []
    for agent in agents:
        if hasattr(agent, "astep"):
            tasks.append(agent.astep())
        elif hasattr(agent, "step"):
            tasks.append(_sync_step(agent))
    await asyncio.gather(*tasks)


async def _sync_step(agent: Agent) -> None:
    """Run synchronous step in async context."""
    agent.step()


def step_agents_multithreaded(agents: list[Agent | LLMAgent]) -> None:
    """Step all agents in parallel using threads."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for agent in agents:
            if hasattr(agent, "step"):
                futures.append(executor.submit(agent.step))
            elif hasattr(agent, "astep"):
                # Optionally, run async steps in the event loop in a thread
                futures.append(
                    executor.submit(lambda agent=agent: asyncio.run(agent.astep()))
                )
        for future in futures:
            future.result()


def step_agents_parallel_sync(agents: list[Agent | LLMAgent]) -> None:
    """Synchronous wrapper for parallel stepping using the global mode."""
    if _PARALLEL_STEPPING_MODE == "asyncio":
        try:
            asyncio.get_running_loop()
            # If in event loop, use thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(step_agents_parallel(agents))
                )
                future.result()
        except RuntimeError:
            # No event loop - create one
            asyncio.run(step_agents_parallel(agents))
    elif _PARALLEL_STEPPING_MODE == "threading":
        step_agents_multithreaded(agents)
    else:
        raise ValueError(f"Unknown parallel stepping mode: {_PARALLEL_STEPPING_MODE}")


# Patch Mesa's shuffle_do for automatic parallel detection
_original_shuffle_do = AgentSet.shuffle_do


def _enhanced_shuffle_do(self, method: str, *args, **kwargs):
    """Enhanced shuffle_do with automatic parallel stepping."""
    if method == "step" and self:
        agent = next(iter(self))
        if hasattr(agent, "model") and getattr(agent.model, "parallel_stepping", False):
            step_agents_parallel_sync(list(self))
            return
    _original_shuffle_do(self, method, *args, **kwargs)


def enable_automatic_parallel_stepping(mode: str = "asyncio"):
    """Enable automatic parallel stepping with selectable mode ('asyncio' or 'threading')."""
    global _PARALLEL_STEPPING_MODE  # noqa: PLW0603
    if mode not in ("asyncio", "threading"):
        raise ValueError("mode must be either 'asyncio' or 'threading'")
    _PARALLEL_STEPPING_MODE = mode
    AgentSet.shuffle_do = _enhanced_shuffle_do


def disable_automatic_parallel_stepping():
    """Restore original shuffle_do behavior."""
    AgentSet.shuffle_do = _original_shuffle_do
