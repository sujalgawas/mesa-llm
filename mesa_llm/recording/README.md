# Mesa-LLM Recording System

Record and analyze agent behavior in mesa-llm simulations.

## Quick Start

```python
from mesa_llm.recording import SimulationRecorder, setup_recording_model

# Method 1: Auto-setup
recorder = setup_recording_model(your_model)
your_model.run(steps=10)
recorder.save()

# Method 2: Manual setup
from mesa_llm.recording import RecordingMixin

class YourModel(RecordingMixin, mesa.Model):
    def __init__(self):
        super().__init__()
        self.setup_recording()
```

## Components

- **`SimulationRecorder`**: Core recording functionality
- **`SimulationAnalyzer`**: Statistical analysis and visualizations
- **`AgentViewer`**: Interactive agent exploration
- **`RecordingMixin`**: Non-invasive model integration

## Analysis

```python
from mesa_llm.recording import SimulationAnalyzer

analyzer = SimulationAnalyzer("recording.json")
stats = analyzer.get_basic_stats()
analyzer.plot_activity_timeline()
```

## Agent Exploration

```python
from mesa_llm.recording import AgentViewer

viewer = AgentViewer("recording.json")
viewer.list_agents()
viewer.view_agent_timeline(agent_id=0)
viewer.interactive_mode()
```

## CLI Tools

```bash
python view_agents.py              # Interactive agent explorer
python quick_agent_demo.py         # Quick agent overview
```

## Output Formats

- **JSON**: Human-readable, 25KB typical
- **Pickle**: Fast loading, 10KB typical
- **Charts**: Activity timelines and network graphs