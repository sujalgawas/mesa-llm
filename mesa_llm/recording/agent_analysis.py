"""
Simple agent viewer for recorded mesa-llm simulations with rich formatting.
"""

import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


class AgentViewer:
    """Simple viewer for exploring agent behavior in recorded simulations."""

    def __init__(self, recording_path: str):
        self.recording_path = Path(recording_path)
        self.data = self._load_recording()
        self.events = self.data["events"]
        self.agent_events = self._organize_events_by_agent()
        self.console = Console()

    def _load_recording(self):
        """Load simulation recording from file."""
        if self.recording_path.suffix == ".pkl":
            with open(self.recording_path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        else:
            with open(self.recording_path) as f:
                return json.load(f)

    def _organize_events_by_agent(self):
        """Organize events by agent ID."""
        agent_events = defaultdict(list)
        for event in self.events:
            if event.get("agent_id") is not None:
                agent_events[event["agent_id"]].append(event)

        # Sort by timestamp
        for agent_id in agent_events:
            agent_events[agent_id].sort(key=lambda x: x["timestamp"])

        return dict(agent_events)

    def _format_event(self, event):
        """Format event content for rich display."""
        content = event["content"]
        event_type = event["event_type"]

        if event_type == "message":
            msg = content.get("message", "")
            recipients = content.get("recipient_ids", [])
            return f"MESSAGE to {recipients}: {msg}"

        elif event_type == "observation":
            lines = ["OBSERVATION"]
            if isinstance(content, dict) and "self_state" in content:
                self_state = content["self_state"]
                lines.append(f"Position: {self_state.get('location', 'Unknown')}")
                if "internal_state" in self_state:
                    lines.append(
                        f"Internal State: {', '.join(map(str, self_state['internal_state']))}"
                    )
            else:
                lines.append(str(content))
            return "\n".join(lines)

        elif event_type == "plan":
            lines = ["PLANNING"]
            if isinstance(content, dict) and "plan_content" in content:
                plan = content["plan_content"].get("content", "")
                lines.append(f"Reasoning: {plan}")
            else:
                lines.append(str(content))
            return "\n".join(lines)

        elif event_type == "action":
            action = (
                content.get("action_type", "")
                if isinstance(content, dict)
                else str(content)
            )
            return f"ACTION: {action}"
        else:
            return f"{event_type.upper()}: {content}"

    def list_agents(self):
        """Show all agents."""
        self.console.print("\nAvailable Agents", style="bold blue")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent ID", style="dim", width=12)
        table.add_column("Total Events", justify="right")
        table.add_column("Event Types", style="green")

        for agent_id in sorted(self.agent_events.keys()):
            events = self.agent_events[agent_id]
            event_types = {e["event_type"] for e in events}
            table.add_row(
                str(agent_id), str(len(events)), ", ".join(sorted(event_types))
            )

        self.console.print(table)

    ############################### displaying of agent events ##################################

    def view_agent_timeline(self, agent_id):
        """Show agent timeline."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        events = self.agent_events[agent_id]
        self.console.print(f"\nTimeline for Agent {agent_id}", style="bold blue")
        self.console.print(f"Showing {len(events)} events\n", style="dim")

        for event in events:
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
            title = (
                f"Step {event['step']} | {timestamp} | {event['event_type'].title()}"
            )
            formatted = self._format_event(event)

            panel = Panel(
                formatted,
                title=title,
                title_align="left",
                border_style="bright_blue"
                if event["event_type"] == "message"
                else "white",
            )
            self.console.print(panel)

    def view_agent_conversations(self, agent_id):
        """Show agent conversations."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        # Get sent and received messages
        sent_messages = [
            e for e in self.agent_events[agent_id] if e["event_type"] == "message"
        ]

        received_messages = []
        for other_id, other_events in self.agent_events.items():
            if other_id == agent_id:
                continue
            for event in other_events:
                if event["event_type"] == "message" and agent_id in event[
                    "content"
                ].get("recipient_ids", []):
                    received_messages.append((other_id, event))

        self.console.print(f"\nConversations for Agent {agent_id}", style="bold blue")

        if not sent_messages and not received_messages:
            self.console.print("No conversations found for this agent.", style="yellow")
            return

        # Combine and sort by timestamp
        all_messages = []
        for msg in sent_messages:
            all_messages.append(("SENT", agent_id, msg))
        for sender_id, msg in received_messages:
            all_messages.append(("RECEIVED", sender_id, msg))

        all_messages.sort(key=lambda x: x[2]["timestamp"])

        for direction, sender_id, event in all_messages:
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
            message = event["content"].get("message", "")

            if direction == "SENT":
                recipients = event["content"].get("recipient_ids", [])
                content = f"To agents {recipients}: {message}"
                title = f"SENT Step {event['step']} | {timestamp}"
                style = "green"
            else:
                content = f"From agent {sender_id}: {message}"
                title = f"RECEIVED Step {event['step']} | {timestamp}"
                style = "blue"

            panel = Panel(content, title=title, title_align="left", border_style=style)
            self.console.print(panel)

    def view_agent_decisions(self, agent_id):
        """Show agent decision-making process."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        events = self.agent_events[agent_id]
        decision_events = [
            e for e in events if e["event_type"] in ["observation", "plan", "action"]
        ]

        self.console.print(f"\nDecision-Making for Agent {agent_id}", style="bold blue")

        # Group by step
        steps = defaultdict(list)
        for event in decision_events:
            steps[event["step"]].append(event)

        for step in sorted(steps.keys()):
            self.console.print(f"\nStep {step} Decision Cycle", style="bold yellow")
            step_events = sorted(
                steps[step],
                key=lambda x: ["observation", "plan", "action"].index(x["event_type"]),
            )
            for event in step_events:
                formatted = self._format_event(event)
                panel = Panel(
                    formatted,
                    title=f"{event['event_type'].title()}",
                    border_style="cyan",
                )
                self.console.print(panel)

    def view_agent_summary(self, agent_id):
        """Show agent summary."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        events = self.agent_events[agent_id]
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event["event_type"]] += 1

        # Count received messages
        received_count = 0
        for other_id, other_events in self.agent_events.items():
            if other_id == agent_id:
                continue
            for event in other_events:
                if event["event_type"] == "message" and agent_id in event[
                    "content"
                ].get("recipient_ids", []):
                    received_count += 1

        self.console.print(f"\nAgent {agent_id} Summary", style="bold blue")

        # Statistics table
        table = Table(title="Activity Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Events", str(len(events)))
        table.add_row("Messages Sent", str(event_counts["message"]))
        table.add_row("Messages Received", str(received_count))
        table.add_row("Observations", str(event_counts["observation"]))
        table.add_row("Plans", str(event_counts["plan"]))
        table.add_row("Actions", str(event_counts["action"]))

        self.console.print(table)

    def interactive_mode(self):
        """Interactive mode for exploring agents."""
        self.console.print("Welcome to the Mesa-LLM Agent Viewer!", style="bold green")
        self.console.print(
            "Explore individual agent behavior from your recorded simulation.\n"
        )

        commands = {
            "list": "Show all agents",
            "timeline": "View agent timeline",
            "conversations": "View agent conversations",
            "decisions": "View agent decision-making",
            "summary": "View agent summary",
            "quit": "Exit viewer",
        }

        while True:
            self.console.print("\nAvailable Commands:", style="bold blue")
            for command, description in commands.items():
                self.console.print(f"• {command} - {description}")

            command = Prompt.ask("\nEnter command").strip().lower()

            if command in ["quit", "q"]:
                self.console.print("Goodbye!", style="yellow")
                break
            elif command == "list":
                self.list_agents()
            else:
                parts = command.split()
                if len(parts) >= 2:
                    try:
                        agent_id = int(parts[1])
                        cmd = parts[0]
                        if cmd == "timeline":
                            self.view_agent_timeline(agent_id)
                        elif cmd == "conversations":
                            self.view_agent_conversations(agent_id)
                        elif cmd == "decisions":
                            self.view_agent_decisions(agent_id)
                        elif cmd == "summary":
                            self.view_agent_summary(agent_id)
                        else:
                            self.console.print(f"Unknown command: {cmd}", style="red")
                    except ValueError:
                        self.console.print(
                            "Invalid agent ID. Please enter a number.", style="red"
                        )
                else:
                    self.console.print("Usage: <command> <agent_id>", style="red")


def quick_agent_view(recording_path: str, agent_id: int, view_type: str = "summary"):
    """Quick view of a specific agent."""
    viewer = AgentViewer(recording_path)

    if view_type == "timeline":
        viewer.view_agent_timeline(agent_id)
    elif view_type == "conversations":
        viewer.view_agent_conversations(agent_id)
    elif view_type == "decisions":
        viewer.view_agent_decisions(agent_id)
    else:
        viewer.view_agent_summary(agent_id)


if __name__ == "__main__":
    path = "recordings/simulation_a22bf17c_20250613_115437.json"
    viewer = AgentViewer(path)
    viewer.interactive_mode()
