# Mesa-LLM Recording Module

A comprehensive recording system for Mesa-LLM simulations that captures agent behavior, interactions, and decision-making processes.

## What it does

The recording module automatically captures:
- **Agent observations** - What agents see in their environment
- **Agent plans** - How agents reason and make decisions
- **Agent actions** - What agents do each step
- **Messages** - Communication between agents
- **State changes** - How agent internal states evolve over time

## Quick Start

### 1. Record a simulation

Use the `@record_model` decorator on your Mesa model:

```python
from mesa_llm.recording import record_model

@record_model
class MySimulation(Model):
    def __init__(self, ...):
        # Your model setup
        pass
```

The decorator automatically:
- Creates a `SimulationRecorder` instance
- Attaches it to all LLM agents
- Records events during simulation steps
- Saves recordings when the simulation ends

The decorator is a class decorator that instruments a Mesa `Model` subclass with a :class:`SimulationRecorder`.


Extra keyword arguments are forwarded to :class:`SimulationRecorder` when it is created.  This allows callers to customise output directory or disable certain event types:

    @record_model(recorder_kwargs={"output_dir": "my_runs", "auto_save_interval": 100})
    class MyModel(Model):
        ...

### 2. View recorded data

Use the `AgentViewer` to explore agent behavior:

```python
from mesa_llm.recording import AgentViewer

# Load a recording file
viewer = AgentViewer("recordings/simulation_abc123.json")

# List all agents
viewer.list_agents()

# View specific agent timeline
viewer.view_agent_timeline(agent_id=1)

# See agent conversations
viewer.view_agent_conversations(agent_id=1)

# Interactive exploration
viewer.interactive_mode()
```

## Configuration

Customize recording behavior:

```python
@record_model(recorder_kwargs={
    "output_dir": "my_recordings",
    "record_observations": True,
    "record_plans": True,
    "record_actions": True,
    "record_messages": True,
    "record_state_changes": False,  # Disable state tracking
    "auto_save_interval": 100       # Save every 100 events
})
class MySimulation(Model):
    pass
```

## Files

- `simulation_recorder.py` - Core recording functionality
- `agent_analysis.py` - Tools for viewing and analyzing recordings
- `integration_hooks.py` - Decorator for easy model integration
- `__init__.py` - Module exports

## Output

Recordings are saved as JSON or Pickle files containing:
- Timestamped events for each agent
- Simulation metadata
- Event statistics
- Communication networks

