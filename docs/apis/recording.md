# Recording Module

The recording system in Mesa-LLM provides comprehensive tools for capturing, analyzing, and visualizing simulation events. It enables researchers and developers to record all agent behavior, communications, decisions, and state changes for post-simulation analysis, debugging, and research insights.

## Core Components

### class SimulationRecorder(model, output_dir="recordings", record_state_changes=True, auto_save_interval=None)

Centralized recorder for capturing all simulation events including agent observations, plans, actions, messages, state changes, and model-level transitions.

**Parameters:**
- **model** (*Model*) – Mesa model instance to record
- **output_dir** (*str*) – Directory for saving recordings (default: "recordings")
- **record_state_changes** (*bool*) – Whether to track agent state changes (default: True)
- **auto_save_interval** (*int | None*) – Automatic save frequency in events (default: None)

**Attributes:**
- **model** - Reference to the Mesa model being recorded
- **events** - List of all recorded SimulationEvent objects
- **simulation_id** - Unique identifier for this recording session
- **start_time** - Recording start timestamp
- **simulation_metadata** - Recording metadata and statistics

**Methods:**

**record_event(event_type, content=None, agent_id=None, metadata=None, recipient_ids=None)**
Record a simulation event with specified type and content.

**record_model_event(event_type, content)**
Record model-level events like simulation start/end or step transitions.

**get_agent_events(agent_id)** → *list[SimulationEvent]*
Retrieve all events for a specific agent.

**get_events_by_type(event_type)** → *list[SimulationEvent]*
Get all events of a specific type (observation, plan, action, message, etc.).

**get_events_by_step(step)** → *list[SimulationEvent]*
Get all events from a specific simulation step.

**export_agent_memory(agent_id)** → *dict*
Export complete agent event history and summary statistics.

**save(filename=None, format="json")** → *Path*
Save complete simulation recording in JSON or pickle format.

**get_stats()** → *dict*
Get comprehensive recording statistics and metadata.

---

### @record_model decorator

**@record_model(cls=None, **kwargs)** → *Callable*

Class decorator that automatically instruments Mesa Model subclasses with SimulationRecorder functionality. Provides seamless integration without manual recorder setup.

**Features:**
- Automatically creates and attaches SimulationRecorder after model initialization
- Attaches recorder to all LLMAgent instances in the model
- Wraps model.step() to record step start/end events
- Provides save_recording() convenience method
- Registers automatic save on program exit

**Parameters:**
- **kwargs** - Forwarded to SimulationRecorder constructor for customization

---

### class SimulationEvent

**SimulationEvent(event_id, timestamp, step, agent_id, event_type, content, metadata)**

Dataclass representing a single recorded event in the simulation with complete context and metadata.

**Attributes:**
- **event_id** (*str*) - Unique identifier for this event
- **timestamp** (*datetime*) - UTC timestamp when event occurred
- **step** (*int*) - Simulation step number
- **agent_id** (*int | None*) - Agent associated with event (None for model events)
- **event_type** (*str*) - Type of event (observation, plan, action, message, state_change, etc.)
- **content** (*dict*) - Event-specific data and information
- **metadata** (*dict*) - Additional contextual metadata

---

### class AgentViewer

**AgentViewer(recording_path)**

Interactive analysis tool for exploring recorded simulation data with rich terminal formatting and comprehensive agent behavior insights.

**Methods:**

**show_simulation_info()**
Display simulation metadata, overview, and agent statistics.

**list_agents()**
Show all agents with event counts and types.

**view_agent_timeline(agent_id)**
Display chronological timeline of all agent events.

**view_agent_conversations(agent_id)**
Show sent and received messages with conversation context.

**view_agent_decisions(agent_id)**
Analyze agent decision-making process by step (observation → plan → action).

**view_agent_summary(agent_id)**
Display comprehensive agent statistics and activity summary.

**interactive_mode()**
Launch interactive terminal interface for exploring simulation data.

---

## Usage Examples

### Basic Recording Setup

```python
from mesa_llm.recording.record_model import record_model
from mesa import Model

@record_model(output_dir="my_recordings", auto_save_interval=50)
class MyModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model initialization
        # Recorder automatically attached after __init__

    def step(self):
        super().step()
        # Step events automatically recorded

# Save recording manually
model = MyModel()
model.run_simulation()
model.save_recording("final_simulation.json")
```

### Manual Recorder Integration

```python
from mesa_llm.recording.simulation_recorder import SimulationRecorder
from mesa_llm.llm_agent import LLMAgent

class MyAgent(LLMAgent):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def step(self):
        # Automatic recording of observations, plans, actions
        obs = self.generate_obs()
        plan = self.reasoning.plan(obs=obs)
        self.apply_plan(plan)

class MyModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorder = SimulationRecorder(
            model=self,
            output_dir="recordings",
            auto_save_interval=100
        )

        # Attach to agents
        for agent in self.agents:
            if hasattr(agent, 'recorder'):
                agent.recorder = self.recorder
```

### Custom Event Recording

```python
def step(self):
    # Record custom domain-specific events
    if hasattr(self, 'recorder'):
        self.recorder.record_event(
            event_type="negotiation_result",
            content={
                "participants": [self.unique_id, other_agent.unique_id],
                "outcome": "agreement_reached",
                "terms": self.negotiation_terms
            },
            agent_id=self.unique_id,
            metadata={"negotiation_round": self.round_number}
        )
```

### Analysis and Visualization

```python
from mesa_llm.recording.agent_analysis import AgentViewer, quick_agent_view

# Interactive exploration
viewer = AgentViewer("recordings/simulation_abc123_20240101_120000.json")
viewer.interactive_mode()

# Quick specific views
quick_agent_view("recording.json", agent_id=5, view_type="timeline")
quick_agent_view("recording.json", agent_id=5, view_type="conversations")
quick_agent_view("recording.json", agent_id=5, view_type="decisions")
quick_agent_view("recording.json", view_type="info")  # Simulation overview
```

### Event Type Categories

**Agent Events:**
- **observation** - Environmental perception and state awareness
- **plan** - Reasoning output and decision-making processes
- **action** - Tool execution and environment interaction
- **message** - Agent-to-agent communication
- **state_change** - Internal agent state modifications

**Model Events:**
- **simulation_start** - Recording initialization
- **simulation_end** - Recording completion with status
- **step_start** - Beginning of simulation step
- **step_end** - Completion of simulation step

**Custom Events:**
- Domain-specific events can be recorded with custom event_type strings
- Useful for tracking domain logic, negotiations, transactions, etc.

### Export and Integration

```python
# Export specific agent data
recorder = model.recorder
agent_data = recorder.export_agent_memory(agent_id=1)

# Get recording statistics
stats = recorder.get_stats()
print(f"Total events: {stats['total_events']}")
print(f"Active agents: {stats['unique_agents']}")

# Filter events for analysis
observations = recorder.get_events_by_type("observation")
step_events = recorder.get_events_by_step(10)
```
