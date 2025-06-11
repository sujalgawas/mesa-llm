# Recording System Architecture

Technical overview of the mesa-llm recording system implementation.

## Core Data Structures

### SimulationEvent
```python
@dataclass
class SimulationEvent:
    event_id: str          # Unique identifier
    timestamp: datetime    # UTC timestamp
    step: int             # Simulation step
    agent_id: int | None  # Agent responsible
    event_type: str       # observation, action, message, plan, state_change
    content: dict         # Event-specific data
    metadata: dict        # Additional context
```

### Recording Flow
1. **Event Capture**: Methods wrapped with decorators
2. **Event Processing**: Content extracted and formatted
3. **Storage**: Events appended to in-memory list
4. **Serialization**: JSON/Pickle export with metadata

## Component Architecture

### SimulationRecorder
- **Event Queue**: `List[SimulationEvent]` - chronological event storage
- **Agent State Tracking**: `Dict[agent_id, state]` - monitors changes
- **Auto-save**: Configurable periodic saves
- **Memory Management**: Tracks agent memory snapshots

### Integration Hooks
- **RecordingMixin**: Adds recording to any Mesa model
- **Method Decorators**: `@record_method()` wraps agent methods
- **Content Extractors**: Type-specific formatters for observations/actions
- **Runtime Setup**: `setup_recording_model()` monkey-patches existing models

### Event Types & Content Extraction

**Observation Events**:
```python
content = {
    'step': observation.step,
    'self_state': observation.self_state,
    'local_state': observation.local_state
}
```

**Action Events**:
```python
content = {
    'action_type': action_type,
    'action_details': action_details,
    'tool_responses': tool_responses
}
```

**Message Events**:
```python
content = {
    'message': message_text,
    'recipient_ids': [agent_ids],
    'sender_id': sender_id
}
```

## Analysis Pipeline

### SimulationAnalyzer
- **Event Filtering**: Query by agent, type, time range
- **Network Analysis**: NetworkX graph of agent communications
- **Temporal Patterns**: Pandas DataFrame for time-series analysis
- **Visualization**: Matplotlib/Seaborn charts

### AgentViewer
- **Event Organization**: `Dict[agent_id, List[AgentEvent]]`
- **Content Formatting**: Rich markup for terminal display
- **Interactive CLI**: Command pattern with prompt loop
- **View Strategies**: Timeline, conversation threads, decision cycles

## Storage Format

### JSON Structure
```json
{
  "metadata": {
    "simulation_id": "abc123",
    "start_time": "2024-01-01T00:00:00Z",
    "total_events": 150,
    "agents_summary": {...}
  },
  "events": [
    {
      "event_id": "evt_001",
      "timestamp": "2024-01-01T00:00:01Z",
      "step": 1,
      "agent_id": 0,
      "event_type": "observation",
      "content": {...},
      "metadata": {...}
    }
  ]
}
```

## Performance Characteristics

- **Memory**: O(n) where n = total events
- **Recording Overhead**: ~5-10% simulation slowdown
- **Storage**: ~1KB per event (JSON), ~400B (Pickle)
- **Query Performance**: O(n) linear scan, O(log n) with indexing

## Extension Points

### Custom Event Types
1. Add content extractor function
2. Register with `@record_method(content_extractor=custom_extractor)`
3. Add formatter to AgentViewer

### Custom Analysis
1. Inherit from `SimulationAnalyzer`
2. Override query methods
3. Add visualization functions

### Integration Patterns
1. **Decorator**: Wrap existing methods
2. **Mixin**: Inherit recording capabilities
3. **Composition**: Inject recorder instance
4. **Monkey-patch**: Runtime modification