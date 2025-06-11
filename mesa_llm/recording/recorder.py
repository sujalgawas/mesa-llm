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
        content: dict[str, Any],
        agent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a simulation event."""
        event_id = f"{self.simulation_id}_{len(self.events):06d}"

        event = SimulationEvent(
            event_id=event_id,
            timestamp=datetime.now(UTC),
            step=self.model.steps,
            agent_id=agent_id,
            event_type=event_type,
            content=content,
            metadata=metadata or {},
        )

        self.events.append(event)
        self.events_since_save += 1

        # Auto-save if configured
        if (
            self.auto_save_interval
            and self.events_since_save >= self.auto_save_interval
        ):
            self.auto_save()

    def record_observation(self, agent_id: int, observation_data: dict[str, Any]):
        """Record an agent observation."""
        if not self.record_observations:
            return

        self.record_event(
            event_type="observation",
            content=observation_data,
            agent_id=agent_id,
            metadata={"source": "agent_observation"},
        )

    def record_plan(self, agent_id: int, plan_data: dict[str, Any]):
        """Record an agent plan."""
        if not self.record_plans:
            return

        self.record_event(
            event_type="plan",
            content=plan_data,
            agent_id=agent_id,
            metadata={"source": "agent_planning"},
        )

    def record_action(self, agent_id: int, action_data: dict[str, Any]):
        """Record an agent action."""
        if not self.record_actions:
            return

        self.record_event(
            event_type="action",
            content=action_data,
            agent_id=agent_id,
            metadata={"source": "agent_action"},
        )

    def record_message(
        self, sender_id: int, message: str, recipient_ids: list[int] | None = None
    ):
        """Record an inter-agent message."""
        if not self.record_messages:
            return

        content = {
            "message": message,
            "recipient_ids": recipient_ids or [],
        }

        self.record_event(
            event_type="message",
            content=content,
            agent_id=sender_id,
            metadata={"source": "agent_communication"},
        )

    def record_state_change(self, agent_id: int, state_changes: dict[str, dict]):
        """Record agent state changes."""
        if not self.record_state_changes:
            return

        self.record_event(
            event_type="state_change",
            content=state_changes,
            agent_id=agent_id,
            metadata={"source": "state_tracking"},
        )

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

    def save(self, filename: str | None = None):
        """Save complete simulation recording."""
        if filename is None:
            filename = f"simulation_{self.simulation_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        # Update metadata
        self.simulation_metadata.update(
            {
                "end_time": datetime.now(UTC).isoformat(),
                "total_steps": self.model.steps,
                "total_events": len(self.events),
                "total_agents": len(self.model.agents),
                "duration_seconds": (
                    datetime.now(UTC) - self.start_time
                ).total_seconds(),
            }
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

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        return filepath

    def save_pickle(self, filename: str | None = None):
        """Save recording in pickle format for faster loading."""
        if filename is None:
            filename = f"simulation_{self.simulation_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pkl"

        filepath = self.output_dir / filename

        # Update metadata
        self.simulation_metadata.update(
            {
                "end_time": datetime.now(UTC).isoformat(),
                "total_steps": self.model.steps,
                "total_events": len(self.events),
                "total_agents": len(self.model.agents),
                "duration_seconds": (
                    datetime.now(UTC) - self.start_time
                ).total_seconds(),
            }
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

        with open(filepath, "wb") as f:
            pickle.dump(export_data, f)

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
