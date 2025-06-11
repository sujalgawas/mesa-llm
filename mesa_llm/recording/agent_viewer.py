"""
User-friendly agent viewer for recorded mesa-llm simulations.

This module provides interactive tools to explore individual agent behavior,
conversations, state changes, and decision-making processes from recorded simulations.
"""

import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class AgentEvent:
    """A processed event specific to an agent."""

    event_id: str
    timestamp: datetime
    step: int
    event_type: str
    content: Any
    metadata: dict[str, Any]
    formatted_content: str


class AgentViewer:
    """
    Interactive viewer for exploring individual agent behavior in recorded simulations.

    Features:
    - Agent-specific timeline views
    - Conversation threads
    - Internal state evolution
    - Decision-making analysis
    - Rich formatted output (if available) or plain text fallback
    """

    def __init__(self, recording_path: str):
        self.recording_path = Path(recording_path)
        self.data = self._load_recording()
        self.events = self.data["events"]
        self.metadata = self.data["metadata"]

        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

        # Process events by agent
        self.agent_events = self._organize_events_by_agent()
        self.agent_ids = list(self.agent_events.keys())

    def _load_recording(self) -> dict[str, Any]:
        """Load simulation recording from file."""
        if self.recording_path.suffix == ".pkl":
            with open(self.recording_path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        else:
            with open(self.recording_path) as f:
                return json.load(f)

    def _organize_events_by_agent(self) -> dict[int, list[AgentEvent]]:
        """Organize events by agent ID with formatted content."""
        agent_events = defaultdict(list)

        for event in self.events:
            agent_id = event.get("agent_id")
            if agent_id is not None:
                formatted_content = self._format_event_content(event)
                agent_event = AgentEvent(
                    event_id=event["event_id"],
                    timestamp=datetime.fromisoformat(event["timestamp"]),
                    step=event["step"],
                    event_type=event["event_type"],
                    content=event["content"],
                    metadata=event["metadata"],
                    formatted_content=formatted_content,
                )
                agent_events[agent_id].append(agent_event)

        # Sort events by timestamp for each agent
        for agent_id in agent_events:
            agent_events[agent_id].sort(key=lambda x: x.timestamp)

        return dict(agent_events)

    def _format_event_content(self, event: dict[str, Any]) -> str:
        """Format event content for display."""
        event_type = event["event_type"]
        content = event["content"]

        if event_type == "observation":
            return self._format_observation(content)
        elif event_type == "plan":
            return self._format_plan(content)
        elif event_type == "message":
            return self._format_message(content)
        elif event_type == "action":
            return self._format_action(content)
        elif event_type == "state_change":
            return self._format_state_change(content)
        else:
            return str(content)

    def _format_observation(self, content: dict[str, Any]) -> str:
        """Format observation content."""
        lines = ["👁️  OBSERVATION"]

        if "self_state" in content:
            self_state = content["self_state"]
            lines.append(f"Position: {self_state.get('location', 'Unknown')}")
            if "internal_state" in self_state:
                lines.append(
                    f"Internal State: {', '.join(map(str, self_state['internal_state']))}"
                )

        if content.get("local_state"):
            lines.append("Nearby Agents:")
            for agent_name, agent_info in content["local_state"].items():
                lines.append(
                    f"  - {agent_name}: {agent_info.get('position', 'Unknown position')}"
                )

        return "\n".join(lines)

    def _format_plan(self, content: dict[str, Any]) -> str:
        """Format plan content."""
        lines = ["🧠 PLANNING"]

        if "plan_content" in content:
            plan_content = content["plan_content"]
            if "content" in plan_content:
                # Extract reasoning from plan content
                reasoning = str(plan_content["content"])
                lines.append(f"Reasoning: {reasoning}")

            if "tool_calls" in plan_content:
                lines.append("Planned Actions:")
                for tool_call in plan_content["tool_calls"]:
                    func_name = tool_call.get("function", "unknown")
                    args = tool_call.get("arguments", "{}")
                    lines.append(f"  - {func_name}({args})")

        return "\n".join(lines)

    def _format_message(self, content: dict[str, Any]) -> str:
        """Format message content."""
        message = content.get("message", "Unknown message")
        recipients = content.get("recipient_ids", [])

        return f'💬 MESSAGE to agents {recipients}: "{message}"'

    def _format_action(self, content: dict[str, Any]) -> str:
        """Format action content."""
        action_type = content.get("action_type", "unknown")
        details = content.get("action_details", {})

        lines = [f"⚡ ACTION: {action_type.upper()}"]
        for key, value in details.items():
            lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    def _format_state_change(self, content: dict[str, Any]) -> str:
        """Format state change content."""
        lines = ["🔄 STATE CHANGE"]

        for field, change in content.items():
            old_val = change.get("old", "None")
            new_val = change.get("new", "None")
            lines.append(f"  - {field}: {old_val} → {new_val}")

        return "\n".join(lines)

    def _print(self, content, style=None):
        """Print with rich formatting if available, otherwise plain text."""
        if self.console:
            self.console.print(content, style=style)
        else:
            # Strip rich markup for plain text
            if hasattr(content, "__rich__"):
                content = str(content)
            # Basic cleanup of rich markup
            content = re.sub(r"\[.*?\]", "", str(content))
            print(content)

    def list_agents(self):
        """Display a list of all agents in the simulation."""
        self._print("\nAvailable Agents", "bold blue")

        if RICH_AVAILABLE and self.console:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Agent ID", style="dim", width=12)
            table.add_column("Total Events", justify="right")
            table.add_column("Event Types", style="green")
            table.add_column("Active Steps", justify="right")

            for agent_id in sorted(self.agent_ids):
                events = self.agent_events[agent_id]
                event_types = {event.event_type for event in events}
                steps = [event.step for event in events]

                table.add_row(
                    str(agent_id),
                    str(len(events)),
                    ", ".join(sorted(event_types)),
                    f"{min(steps)}-{max(steps)}" if steps else "None",
                )

            self.console.print(table)
        else:
            # Plain text fallback
            print(f"{'Agent ID':<10} {'Events':<8} {'Event Types':<30} {'Steps'}")
            print("-" * 60)
            for agent_id in sorted(self.agent_ids):
                events = self.agent_events[agent_id]
                event_types = {event.event_type for event in events}
                steps = [event.step for event in events]

                print(
                    f"{agent_id:<10} {len(events):<8} {', '.join(sorted(event_types)):<30} "
                    f"{min(steps)}-{max(steps)}"
                    if steps
                    else "None"
                )

    def view_agent_timeline(self, agent_id: int, event_types: list[str] | None = None):
        """Display a complete timeline for a specific agent."""
        if agent_id not in self.agent_events:
            self._print(f"Agent {agent_id} not found in recording.", "red")
            return

        events = self.agent_events[agent_id]

        # Filter by event types if specified
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        self._print(f"\nTimeline for Agent {agent_id}", "bold blue")
        self._print(f"Showing {len(events)} events\n", "dim")

        for event in events:
            timestamp_str = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
            title = f"Step {event.step} | {timestamp_str} | {event.event_type.title()}"

            if RICH_AVAILABLE and self.console:
                panel = Panel(
                    event.formatted_content,
                    title=title,
                    title_align="left",
                    border_style="bright_blue"
                    if event.event_type == "message"
                    else "white",
                )
                self.console.print(panel)
            else:
                print(f"\n=== {title} ===")
                print(event.formatted_content)
                print("-" * 50)

    def view_agent_conversations(self, agent_id: int):
        """Display all conversations involving a specific agent."""
        if agent_id not in self.agent_events:
            self._print(f"Agent {agent_id} not found in recording.", "red")
            return

        # Get all message events for this agent
        message_events = [
            e for e in self.agent_events[agent_id] if e.event_type == "message"
        ]

        # Also get messages sent TO this agent from other agents
        received_messages = []
        for other_agent_id, other_events in self.agent_events.items():
            if other_agent_id == agent_id:
                continue
            for event in other_events:
                if (
                    event.event_type == "message"
                    and "recipient_ids" in event.content
                    and agent_id in event.content["recipient_ids"]
                ):
                    received_messages.append((other_agent_id, event))

        self._print(f"\nConversations for Agent {agent_id}", "bold blue")

        if not message_events and not received_messages:
            self._print("No conversations found for this agent.", "yellow")
            return

        # Combine and sort all messages by timestamp
        all_messages = []

        # Add sent messages
        for event in message_events:
            all_messages.append(("sent", agent_id, event))

        # Add received messages
        for sender_id, event in received_messages:
            all_messages.append(("received", sender_id, event))

        all_messages.sort(key=lambda x: x[2].timestamp)

        # Display conversation thread
        for direction, sender_id, event in all_messages:
            message = event.content.get("message", "Unknown message")
            recipients = event.content.get("recipient_ids", [])
            timestamp_str = event.timestamp.strftime("%H:%M:%S")

            if direction == "sent":
                title = f"SENT Step {event.step} | {timestamp_str}"
                content = f"To agents {recipients}: {message}"
                style = "green"
            else:
                title = f"RECEIVED Step {event.step} | {timestamp_str}"
                content = f"From agent {sender_id}: {message}"
                style = "blue"

            if RICH_AVAILABLE and self.console:
                panel = Panel(
                    content, title=title, title_align="left", border_style=style
                )
                self.console.print(panel)
            else:
                print(f"\n=== {title} ===")
                print(content)
                print("-" * 40)

    def view_agent_decision_making(self, agent_id: int):
        """Display the decision-making process for a specific agent."""
        if agent_id not in self.agent_events:
            self._print(f"Agent {agent_id} not found in recording.", "red")
            return

        events = self.agent_events[agent_id]

        # Group events by step to show decision cycles
        steps = defaultdict(list)
        for event in events:
            if event.event_type in ["observation", "plan", "action"]:
                steps[event.step].append(event)

        self._print(f"\nDecision-Making Process for Agent {agent_id}", "bold blue")

        for step in sorted(steps.keys()):
            step_events = sorted(
                steps[step],
                key=lambda x: ["observation", "plan", "action"].index(x.event_type),
            )

            self._print(f"\nStep {step} Decision Cycle", "bold yellow")

            for event in step_events:
                if RICH_AVAILABLE and self.console:
                    panel = Panel(
                        event.formatted_content,
                        title=f"{event.event_type.title()}",
                        border_style="cyan",
                    )
                    self.console.print(panel)
                else:
                    print(f"\n--- {event.event_type.title()} ---")
                    print(event.formatted_content)

    def view_agent_summary(self, agent_id: int):
        """Display a comprehensive summary for a specific agent."""
        if agent_id not in self.agent_events:
            self._print(f"Agent {agent_id} not found in recording.", "red")
            return

        events = self.agent_events[agent_id]

        # Calculate statistics
        event_counts = defaultdict(int)
        steps_active = set()
        messages_sent = 0
        messages_received = 0

        for event in events:
            event_counts[event.event_type] += 1
            steps_active.add(event.step)
            if event.event_type == "message":
                messages_sent += 1

        # Count received messages
        for other_agent_id, other_events in self.agent_events.items():
            if other_agent_id == agent_id:
                continue
            for event in other_events:
                if (
                    event.event_type == "message"
                    and "recipient_ids" in event.content
                    and agent_id in event.content["recipient_ids"]
                ):
                    messages_received += 1

        self._print(f"\nAgent {agent_id} Summary", "bold blue")

        if RICH_AVAILABLE and self.console:
            # Statistics table
            stats_table = Table(title="Activity Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")

            stats_table.add_row("Total Events", str(len(events)))
            stats_table.add_row(
                "Active Steps",
                f"{min(steps_active)}-{max(steps_active)}" if steps_active else "None",
            )
            stats_table.add_row("Messages Sent", str(messages_sent))
            stats_table.add_row("Messages Received", str(messages_received))
            stats_table.add_row("Observations Made", str(event_counts["observation"]))
            stats_table.add_row("Plans Created", str(event_counts["plan"]))
            stats_table.add_row("Actions Taken", str(event_counts["action"]))

            self.console.print(stats_table)

            # Activity breakdown
            activity_table = Table(title="Event Type Breakdown")
            activity_table.add_column("Event Type", style="cyan")
            activity_table.add_column("Count", style="green", justify="right")
            activity_table.add_column("Percentage", style="yellow", justify="right")

            total_events = len(events)
            for event_type, count in sorted(event_counts.items()):
                percentage = (count / total_events) * 100
                activity_table.add_row(
                    event_type.title(), str(count), f"{percentage:.1f}%"
                )

            self.console.print(activity_table)
        else:
            # Plain text summary
            print("\nActivity Statistics:")
            print(f"Total Events: {len(events)}")
            print(
                f"Active Steps: {min(steps_active)}-{max(steps_active)}"
                if steps_active
                else "None"
            )
            print(f"Messages Sent: {messages_sent}")
            print(f"Messages Received: {messages_received}")
            print(f"Observations Made: {event_counts['observation']}")
            print(f"Plans Created: {event_counts['plan']}")
            print(f"Actions Taken: {event_counts['action']}")

            print("\nEvent Type Breakdown:")
            total_events = len(events)
            for event_type, count in sorted(event_counts.items()):
                percentage = (count / total_events) * 100
                print(f"{event_type.title()}: {count} ({percentage:.1f}%)")

    def interactive_mode(self):
        """Start interactive mode for exploring agents."""
        self._print("Welcome to the Mesa-LLM Agent Viewer!", "bold green")
        self._print(
            "Explore individual agent behavior from your recorded simulation.\n"
        )

        while True:
            self._print("\nAvailable Commands:", "bold blue")
            print("1. list - Show all agents")
            print("2. timeline <agent_id> - View agent timeline")
            print("3. conversations <agent_id> - View agent conversations")
            print("4. decisions <agent_id> - View agent decision-making")
            print("5. summary <agent_id> - View agent summary")
            print("6. quit - Exit viewer")

            if RICH_AVAILABLE:
                command = Prompt.ask("\nEnter command").strip().lower()
            else:
                command = input("\nEnter command: ").strip().lower()

            if command == "quit" or command == "q":
                self._print("Goodbye!", "yellow")
                break

            elif command == "list":
                self.list_agents()

            elif command.startswith("timeline"):
                parts = command.split()
                if len(parts) >= 2:
                    try:
                        agent_id = int(parts[1])
                        self.view_agent_timeline(agent_id)
                    except ValueError:
                        self._print("Invalid agent ID. Please enter a number.", "red")
                else:
                    self._print("Usage: timeline <agent_id>", "red")

            elif command.startswith("conversations"):
                parts = command.split()
                if len(parts) >= 2:
                    try:
                        agent_id = int(parts[1])
                        self.view_agent_conversations(agent_id)
                    except ValueError:
                        self._print("Invalid agent ID. Please enter a number.", "red")
                else:
                    self._print("Usage: conversations <agent_id>", "red")

            elif command.startswith("decisions"):
                parts = command.split()
                if len(parts) >= 2:
                    try:
                        agent_id = int(parts[1])
                        self.view_agent_decision_making(agent_id)
                    except ValueError:
                        self._print("Invalid agent ID. Please enter a number.", "red")
                else:
                    self._print("Usage: decisions <agent_id>", "red")

            elif command.startswith("summary"):
                parts = command.split()
                if len(parts) >= 2:
                    try:
                        agent_id = int(parts[1])
                        self.view_agent_summary(agent_id)
                    except ValueError:
                        self._print("Invalid agent ID. Please enter a number.", "red")
                else:
                    self._print("Usage: summary <agent_id>", "red")

            else:
                self._print(
                    "Unknown command. Try 'list', 'timeline <id>', 'conversations <id>', 'decisions <id>', 'summary <id>', or 'quit'.",
                    "red",
                )


def create_agent_viewer(recording_path: str) -> AgentViewer:
    """Convenience function to create an agent viewer."""
    return AgentViewer(recording_path)


def quick_agent_view(recording_path: str, agent_id: int, view_type: str = "summary"):
    """Quick view of a specific agent."""
    viewer = AgentViewer(recording_path)

    if view_type == "timeline":
        viewer.view_agent_timeline(agent_id)
    elif view_type == "conversations":
        viewer.view_agent_conversations(agent_id)
    elif view_type == "decisions":
        viewer.view_agent_decision_making(agent_id)
    else:
        viewer.view_agent_summary(agent_id)
