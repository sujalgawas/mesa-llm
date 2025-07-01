import asyncio
import threading
import time

import pytest

from mesa_llm.parallel_stepping import step_agents_parallel, step_agents_parallel_sync

# --- Fixtures for mock agents ---


class SyncAgent:
    def __init__(self):
        self.stepped = False

    def step(self):
        self.stepped = True


class AsyncAgent:
    def __init__(self):
        self.stepped = False

    async def astep(self):
        self.stepped = True


class SleepyAsyncAgent:
    def __init__(self, sleep_time):
        self.stepped = False
        self.sleep_time = sleep_time

    async def astep(self):
        await asyncio.sleep(self.sleep_time)
        self.stepped = True


# --- Tests ---


@pytest.mark.asyncio
async def test_all_sync_agents():
    agents = [SyncAgent() for _ in range(5)]
    await step_agents_parallel(agents)
    assert all(agent.stepped for agent in agents)


@pytest.mark.asyncio
async def test_all_async_agents():
    agents = [AsyncAgent() for _ in range(5)]
    await step_agents_parallel(agents)
    assert all(agent.stepped for agent in agents)


@pytest.mark.asyncio
async def test_mixed_sync_async_agents():
    agents = [SyncAgent(), AsyncAgent(), SyncAgent(), AsyncAgent()]
    await step_agents_parallel(agents)
    assert all(getattr(agent, "stepped", False) for agent in agents)


@pytest.mark.asyncio
async def test_parallelism_with_sleep():
    # If run sequentially, would take 0.2 + 0.3 = 0.5s; in parallel, should be ~0.3s
    agents = [SleepyAsyncAgent(0.2), SleepyAsyncAgent(0.3)]
    start = time.perf_counter()
    await step_agents_parallel(agents)
    elapsed = time.perf_counter() - start
    assert all(agent.stepped for agent in agents)
    assert elapsed < 0.45  # Should be less than sum, allow some overhead


def test_step_agents_parallel_sync_outside_event_loop():
    agents = [SyncAgent() for _ in range(3)]
    step_agents_parallel_sync(agents)
    assert all(agent.stepped for agent in agents)


def test_step_agents_parallel_sync_inside_event_loop():
    agents = [SyncAgent() for _ in range(3)]

    def run():
        step_agents_parallel_sync(agents)

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.get_event_loop().run_in_executor(None, run))
        loop.close()

    t = threading.Thread(target=thread_target)
    t.start()
    t.join()
    assert all(agent.stepped for agent in agents)
