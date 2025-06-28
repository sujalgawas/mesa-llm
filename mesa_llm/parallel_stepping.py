"""
Automatic parallel stepping for Mesa-LLM simulations.
"""

import asyncio

from mesa.agent import Agent, AgentSet

from .llm_agent import LLMAgent


async def step_agents_parallel(agents: list[Agent | LLMAgent]) -> None:
    """Step all agents in parallel using async/await."""
    tasks = []
    for agent in agents:
        if hasattr(agent, "astep"):
            tasks.append(agent.astep())
        else:
            tasks.append(_sync_step(agent))
    await asyncio.gather(*tasks)


async def _sync_step(agent: Agent) -> None:
    """Run synchronous step in async context."""
    if hasattr(agent, "step"):
        agent.step()


def step_agents_parallel_sync(agents: list[Agent | LLMAgent]) -> None:
    """Synchronous wrapper for parallel stepping."""
    try:
        asyncio.get_running_loop()
        # If in event loop, use thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(step_agents_parallel(agents)))
            future.result()
    except RuntimeError:
        # No event loop - create one
        asyncio.run(step_agents_parallel(agents))


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


def enable_automatic_parallel_stepping():
    """Enable automatic parallel stepping."""
    AgentSet.shuffle_do = _enhanced_shuffle_do


def disable_automatic_parallel_stepping():
    """Restore original shuffle_do behavior."""
    AgentSet.shuffle_do = _original_shuffle_do
