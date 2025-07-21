# LLMAgent

LLMAgent is the core agent class in Mesa-LLM that extends Mesa's base Agent class with Large Language Model capabilities. It provides a complete framework for creating intelligent agents that can reason, remember, communicate, and act in simulations using natural language processing.


### class LLMAgent(model, api_key, reasoning, llm_model="gemini/gemini-2.0-flash", system_prompt=None, vision=None, internal_state=None, recorder=None, step_prompt=None)


**Attributes:**
- **model** - Mesa model instance the agent belongs to
- **llm** - ModuleLLM instance for language model communication
- **memory** - Memory instance (default: STLTMemory) for storing experiences
- **tool_manager** - ToolManager for available agent actions
- **reasoning** - Reasoning strategy for decision-making
- **vision** - Perception radius (-1: global, 0: none, >0: local radius)
- **internal_state** - List of internal state attribute strings
- **recorder** - Optional SimulationRecorder for behavior capture

**Parameters:**
- **model** (*Model*) – Mesa model the agent is linked to
- **api_key** (*str*) – API key for LLM provider
- **reasoning** (*type[Reasoning]*) – Reasoning strategy class for decision-making
- **llm_model** (*str*) – Model format 'provider/model', defaults to 'gemini/gemini-2.0-flash'
- **system_prompt** (*str | None*) – Optional system prompt for LLM interactions
- **vision** (*float | None*) – Vision radius for environmental perception
- **internal_state** (*list[str] | str | None*) – Initial internal state attributes
- **recorder** (*SimulationRecorder | None*) – Optional simulation event recorder
- **step_prompt** (*str | None*) – Default prompt for step-based reasoning

**Methods:**

**apply_plan(plan)** → *list[dict]*
Execute plan in simulation environment. Returns tool execution results.

**generate_obs()** → *Observation*
Generate comprehensive observation of current environment and agent state. Scope depends on vision setting.

**send_message(message, recipients)** → *str*
Send message to recipient agents and add to memory. Returns delivery confirmation.

**pre_step()** / **post_step()**
Execute preprocessing/postprocessing around agent step (automatic via __init_subclass__).

**async astep()**
Asynchronous step method for parallel execution. Override for custom async behavior.

### Basic Agent Implementation

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning

class MyAgent(LLMAgent):
    def __init__(self, model, api_key, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key,
            reasoning=CoTReasoning,
            llm_model="openai/gpt-4o",
            system_prompt="You are a helpful agent in a simulation.",
            vision=2,  # See 2 cells in each direction
            internal_state=["curious", "cooperative"],
            step_prompt="Decide what to do next based on your observations."
        )

        # You can override default memory with EpisodicMemory (default is STLTMemory)
        self.memory = EpisodicMemory(
            agent=self,
            api_key=api_key,
            llm_model="openai/gpt-4o-mini",
            max_memory=20
        )

    def step(self):
        # Generate current observation
        obs = self.generate_obs()

        # Use reasoning to create plan
        plan = self.reasoning.plan(obs=obs)

        # Execute the plan
        self.apply_plan(plan)
```


### Parallel Execution example

```python
class ParallelAgent(LLMAgent):
    async def astep(self): # Use 'async' with astep method to enable parallel execution
        """Asynchronous step for parallel execution"""
        obs = self.generate_obs()
        plan = await self.reasoning.aplan( # Use 'await' with aplan method to enable parallel execution
            prompt=self.step_prompt,
            obs=obs
        )
        self.apply_plan(plan)
```

### Agent Communication example

```python
def step(self):
    obs = self.generate_obs()

    # Find nearby agents
    neighbors = [agent for agent in obs.local_state.keys()]
    if neighbors:
        # Send message to neighbors
        neighbor_ids = [int(name.split()[-1]) for name in neighbors]
        self.send_message("Hello neighbors!",
                         [agent for agent in self.model.agents
                          if agent.unique_id in neighbor_ids])

    plan = self.reasoning.plan(obs=obs)
    self.apply_plan(plan)
```


