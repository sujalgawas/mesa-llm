"""
Comprehensive simulation recorder for mesa-llm simulations.

This module provides tools to record all simulation events for post-analysis,
including agent observations, plans, actions, messages, and state changes.
"""

import json
import pickle
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class SimulationEvent:
    """A single recorded event in the simulation."""

    event_id: str
    timestamp: datetime
    step: int
    agent_id: int | None
    event_type: str
    content: dict[str, Any]
    metadata: dict[str, Any]


class SimulationRecorder:
    """
    Centralized recorder for capturing all simulation events for post-analysis.

    This recorder captures:
    - Agent observations and perceptions
    - Agent plans and reasoning processes
    - Agent actions and their outcomes
    - Inter-agent messages and communication
    - Agent state changes over time
    - Model-level events and transitions

    Features:
    - Configurable recording options
    - Automatic event timestamping
    - Memory-efficient batch operations
    - Multiple export formats (JSON, Pickle)
    - Agent state change detection
    - Auto-save functionality for large simulations
    """

    def __init__(
        self,
        model,
        output_dir: str = "recordings",
        record_observations: bool = True,
        record_plans: bool = True,
        record_actions: bool = True,
        record_messages: bool = True,
        record_state_changes: bool = True,
        auto_save_interval: int | None = None,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Recording configuration
        self.record_observations = record_observations
        self.record_plans = record_plans
        self.record_actions = record_actions
        self.record_messages = record_messages
        self.record_state_changes = record_state_changes
        self.auto_save_interval = auto_save_interval

        # Internal state
        self.events: list[SimulationEvent] = []
        self.simulation_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now(UTC)

        # Agent state tracking for change detection
        self.previous_agent_states: dict[int, dict[str, Any]] = {}

        # Auto-save counter
        self.events_since_save = 0

        # Initialize simulation metadata
        self.simulation_metadata = {
            "simulation_id": self.simulation_id,
            "start_time": self.start_time.isoformat(),
            "model_class": self.model.__class__.__name__,
            "recording_config": {
                "observations": record_observations,
                "plans": record_plans,
                "actions": record_actions,
                "messages": record_messages,
                "state_changes": record_state_changes,
            },
        }

    def record_event(
        self,
        event_type: str,
        content: dict[str, Any] | str | None = None,
        agent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        # Legacy parameters from record method for backwards compatibility
        data: dict[str, Any] | str | None = None,
        recipient_ids: list[int] | None = None,
    ):
        """Record a simulation event.

        Args:
            event_type: Type of event to record (observation, plan, action, message, state_change, etc.)
            content: Event content as dict or string (preferred parameter)
            agent_id: ID of the agent associated with this event
            metadata: Additional metadata for the event
            data: Event data (legacy parameter, use content instead)
            recipient_ids: List of recipient IDs for message events
        """
        print(
            f"Recording {event_type} for agent {agent_id} #########################################################"
        )
        # Check if recording is enabled for this event type
        record_config = self.simulation_metadata["recording_config"]

        # Map event types to config keys
        config_key_map = {
            "observation": "observations",
            "plan": "plans",
            "action": "actions",
            "message": "messages",
            "state_change": "state_changes",
        }

        config_key = config_key_map.get(event_type, event_type)
        if not record_config.get(config_key, True):
            return

        # Handle backwards compatibility - if data is provided but content is not, use data
        if content is None and data is not None:
            content = data

        # Handle different content formats based on event type
        if event_type == "message":
            if isinstance(content, str | dict):
                formatted_content = {
                    "message": content,
                    "recipient_ids": recipient_ids or [],
                }
            else:
                formatted_content = {
                    "message": content,
                    "recipient_ids": recipient_ids or [],
                }
        else:
            if isinstance(content, dict):
                formatted_content = content
            else:
                formatted_content = {"data": content}

        # Set metadata source
        source_map = {
            "observation": "agent_observation",
            "plan": "agent_planning",
            "action": "agent_action",
            "message": "agent_communication",
            "state_change": "state_tracking",
        }

        # Merge provided metadata with source metadata
        final_metadata = {"source": source_map.get(event_type, "unknown")}
        if metadata:
            final_metadata.update(metadata)

        # Create the event
        event_id = f"{self.simulation_id}_{len(self.events):06d}"

        event = SimulationEvent(
            event_id=event_id,
            timestamp=datetime.now(UTC),
            step=self.model.steps,
            agent_id=agent_id,
            event_type=event_type,
            content=formatted_content,
            metadata=final_metadata,
        )

        self.events.append(event)
        self.events_since_save += 1

        # Auto-save if configured
        if (
            self.auto_save_interval
            and self.events_since_save >= self.auto_save_interval
        ):
            self.auto_save()

    def track_agent_state(self, agent_id: int, current_state: dict[str, Any]):
        """Track agent state and record changes."""
        if not self.record_state_changes:
            return

        if agent_id in self.previous_agent_states:
            changes = {}
            previous_state = self.previous_agent_states[agent_id]

            for key, new_value in current_state.items():
                old_value = previous_state.get(key)
                if old_value != new_value:
                    changes[key] = {"old": old_value, "new": new_value}

            if changes:
                self.record_state_change(agent_id, changes)

        self.previous_agent_states[agent_id] = current_state.copy()

    def record_model_event(self, event_type: str, content: dict[str, Any]):
        """Record a model-level event."""
        self.record_event(
            event_type=event_type,
            content=content,
            agent_id=None,
            metadata={"source": "model"},
        )

    def get_agent_events(self, agent_id: int) -> list[SimulationEvent]:
        """Get all events for a specific agent."""
        return [event for event in self.events if event.agent_id == agent_id]

    def get_events_by_type(self, event_type: str) -> list[SimulationEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_step(self, step: int) -> list[SimulationEvent]:
        """Get all events from a specific simulation step."""
        return [event for event in self.events if event.step == step]

    def export_agent_memory(self, agent_id: int) -> dict[str, Any]:
        """Export agent memory state for external analysis."""
        agent_events = self.get_agent_events(agent_id)

        return {
            "agent_id": agent_id,
            "events": [asdict(event) for event in agent_events],
            "summary": {
                "total_events": len(agent_events),
                "event_types": list({event.event_type for event in agent_events}),
                "active_steps": list({event.step for event in agent_events}),
                "first_event": (
                    agent_events[0].timestamp.isoformat() if agent_events else None
                ),
                "last_event": (
                    agent_events[-1].timestamp.isoformat() if agent_events else None
                ),
            },
        }

    def get_communication_network(self) -> dict[str, Any]:
        """Analyze communication patterns between agents."""
        message_events = self.get_events_by_type("message")

        # Build communication graph
        communications = defaultdict(lambda: defaultdict(int))
        for event in message_events:
            sender = event.agent_id
            recipients = event.content.get("recipient_ids", [])
            for recipient in recipients:
                communications[sender][recipient] += 1

        return {
            "total_messages": len(message_events),
            "communication_matrix": dict(communications),
            "agents_involved": list(
                set(
                    [event.agent_id for event in message_events]
                    + [
                        r
                        for event in message_events
                        for r in event.content.get("recipient_ids", [])
                    ]
                )
            ),
        }

    def auto_save(self):
        """Automatically save current state."""
        filename = f"autosave_{self.simulation_id}_{len(self.events)}.json"
        self.save(filename)
        self.events_since_save = 0

    def save(self, filename: str | None = None, format: str = "json"):
        """Save complete simulation recording.

        Args:
            filename: Optional filename. If None, auto-generates based on format.
            format: Save format, either "json" or "pickle".
        """
        if format not in ["json", "pickle"]:
            raise ValueError("Format must be 'json' or 'pickle'")

        if filename is None:
            extension = "json" if format == "json" else "pkl"
            filename = f"simulation_{self.simulation_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.{extension}"

        filepath = self.output_dir / filename

        # Update metadata with final state
        self.simulation_metadata.update(
            {
                "end_time": datetime.now(UTC).isoformat(),
                "total_steps": self.model.steps,
                "total_events": len(self.events),
                "total_agents": len(self.model.agents),
                "duration_seconds": (
                    datetime.now(UTC) - self.start_time
                ).total_seconds(),
                # Determine completion status gracefully when `max_steps` is absent
                "completion_status": (
                    "unknown"
                    if getattr(self.model, "max_steps", None) is None
                    else (
                        "interrupted"
                        if self.model.steps < self.model.max_steps
                        else "completed"
                    )
                ),
                "final_step": self.model.steps,
            }
        )

        # Record final model state
        self.record_model_event(
            event_type="simulation_end",
            content={
                "status": (
                    "unknown"
                    if getattr(self.model, "max_steps", None) is None
                    else (
                        "interrupted"
                        if self.model.steps < self.model.max_steps
                        else "completed"
                    )
                ),
                "final_step": self.model.steps,
                "total_events": len(self.events),
            },
        )

        # Prepare export data
        export_data = {
            "metadata": self.simulation_metadata,
            "events": [asdict(event) for event in self.events],
            "agent_summaries": {
                agent_id: self.export_agent_memory(agent_id)["summary"]
                for agent_id in {
                    event.agent_id
                    for event in self.events
                    if event.agent_id is not None
                }
            },
            "communication_network": self.get_communication_network(),
        }

        # Save based on format
        if format == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(export_data, f)

        print(f"Simulation recording saved to: {filepath}")
        return filepath

    def get_stats(self) -> dict[str, Any]:
        """Get recording statistics."""
        agent_ids = {
            event.agent_id for event in self.events if event.agent_id is not None
        }

        return {
            "total_events": len(self.events),
            "unique_agents": len(agent_ids),
            "event_types": list({event.event_type for event in self.events}),
            "simulation_steps": self.model.steps,
            "recording_duration": (datetime.now(UTC) - self.start_time).total_seconds(),
            "events_per_agent": {
                agent_id: len(self.get_agent_events(agent_id)) for agent_id in agent_ids
            },
        }

    def record_observation(self, agent_id: int, content: dict[str, Any]):
        """Record an observation event."""
        self.record_event("observation", content, agent_id)

    def record_plan(self, agent_id: int, content: dict[str, Any]):
        """Record a planning event."""
        self.record_event("plan", content, agent_id)

    def record_action(self, agent_id: int, content: dict[str, Any]):
        """Record an action event."""
        self.record_event("action", content, agent_id)

    def record_message(
        self, agent_id: int, message: str, recipient_ids: list[int] | None = None
    ):
        """Record a message event."""
        self.record_event("message", message, agent_id, recipient_ids=recipient_ids)

    def record_state_change(self, agent_id: int, changes: dict[str, Any]):
        """Record a state change event."""
        self.record_event("state_change", changes, agent_id)
