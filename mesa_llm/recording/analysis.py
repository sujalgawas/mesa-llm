"""
Comprehensive analysis tools for recorded mesa-llm simulations.

This module provides tools for analyzing recorded simulation data including
statistical analysis, visualization, and pattern recognition.
"""

import json
import pickle
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SimulationAnalyzer:
    """
    Comprehensive analyzer for recorded simulation data.

    Provides tools for:
    - Statistical analysis of agent behaviors
    - Communication network analysis
    - Temporal pattern analysis
    - Agent decision-making pattern analysis
    - Cross-simulation comparison
    - Visualization capabilities
    """

    def __init__(self, recording_path: str):
        self.recording_path = Path(recording_path)
        self.data = self._load_recording()
        self.events = self.data["events"]
        self.metadata = self.data["metadata"]

        # Optional dependencies for advanced analysis
        self.has_pandas = self._check_pandas()
        self.has_networkx = self._check_networkx()
        self.has_matplotlib = self._check_matplotlib()
        self.has_seaborn = self._check_seaborn()

    def _load_recording(self) -> dict[str, Any]:
        """Load simulation recording from file."""
        if self.recording_path.suffix == ".pkl":
            with open(self.recording_path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        else:
            with open(self.recording_path) as f:
                return json.load(f)

    def _check_pandas(self) -> bool:
        try:
            import pandas  # noqa: F401

            return True
        except ImportError:
            return False

    def _check_networkx(self) -> bool:
        try:
            import networkx  # noqa: F401

            return True
        except ImportError:
            return False

    def _check_matplotlib(self) -> bool:
        try:
            import matplotlib  # noqa: F401

            return True
        except ImportError:
            return False

    def _check_seaborn(self) -> bool:
        try:
            import seaborn  # noqa: F401

            return True
        except ImportError:
            return False

    def get_basic_stats(self) -> dict[str, Any]:
        """Get basic statistics about the simulation."""
        agents = {
            event.get("agent_id")
            for event in self.events
            if event.get("agent_id") is not None
        }
        event_types = Counter(event["event_type"] for event in self.events)

        return {
            "total_events": len(self.events),
            "unique_agents": len(agents),
            "simulation_steps": max(event["step"] for event in self.events)
            if self.events
            else 0,
            "event_types": dict(event_types),
            "agents_list": sorted(agents),
            "start_time": self.metadata.get("start_time"),
            "end_time": self.metadata.get("end_time"),
        }

    def analyze_agent_activity(self) -> dict[int, dict[str, Any]]:
        """Analyze activity patterns for each agent."""
        agent_activity = defaultdict(
            lambda: {
                "total_events": 0,
                "event_types": Counter(),
                "active_steps": set(),
                "first_event_step": float("inf"),
                "last_event_step": 0,
            }
        )

        for event in self.events:
            agent_id = event.get("agent_id")
            if agent_id is not None:
                activity = agent_activity[agent_id]
                activity["total_events"] += 1
                activity["event_types"][event["event_type"]] += 1
                activity["active_steps"].add(event["step"])
                activity["first_event_step"] = min(
                    activity["first_event_step"], event["step"]
                )
                activity["last_event_step"] = max(
                    activity["last_event_step"], event["step"]
                )

        # Convert sets to lists and fix infinity
        for agent_id in agent_activity:
            activity = agent_activity[agent_id]
            activity["active_steps"] = sorted(activity["active_steps"])
            activity["event_types"] = dict(activity["event_types"])
            if activity["first_event_step"] == float("inf"):
                activity["first_event_step"] = 0

        return dict(agent_activity)

    def analyze_communication_patterns(self) -> dict[str, Any]:
        """Analyze communication patterns between agents."""
        message_events = [e for e in self.events if e["event_type"] == "message"]

        if not message_events:
            return {
                "total_messages": 0,
                "communication_matrix": {},
                "most_active_communicators": [],
            }

        # Build communication matrix
        communication_matrix = defaultdict(lambda: defaultdict(int))
        all_agents = set()

        for event in message_events:
            sender = event.get("agent_id")
            recipients = event.get("content", {}).get("recipient_ids", [])

            if sender is not None:
                all_agents.add(sender)
                for recipient in recipients:
                    all_agents.add(recipient)
                    communication_matrix[sender][recipient] += 1

        # Find most active communicators
        sender_counts = Counter()
        receiver_counts = Counter()

        for sender, recipients in communication_matrix.items():
            sender_counts[sender] = sum(recipients.values())

        for recipients in communication_matrix.values():
            for recipient, count in recipients.items():
                receiver_counts[recipient] += count

        return {
            "total_messages": len(message_events),
            "communication_matrix": {
                k: dict(v) for k, v in communication_matrix.items()
            },
            "most_active_senders": sender_counts.most_common(5),
            "most_active_receivers": receiver_counts.most_common(5),
            "agents_involved": sorted(all_agents),
        }

    def analyze_temporal_patterns(self) -> dict[str, Any]:
        """Analyze temporal patterns in the simulation."""
        if not self.has_pandas:
            return {"error": "Pandas required for temporal analysis"}

        import pandas as pd

        # Convert events to DataFrame
        df_data = []
        for event in self.events:
            df_data.append(
                {
                    "timestamp": pd.to_datetime(event["timestamp"]),
                    "step": event["step"],
                    "agent_id": event.get("agent_id"),
                    "event_type": event["event_type"],
                }
            )

        df = pd.DataFrame(df_data)

        # Analyze activity by step
        step_activity = df.groupby("step").size()
        event_type_by_step = (
            df.groupby(["step", "event_type"]).size().unstack(fill_value=0)
        )

        return {
            "activity_by_step": step_activity.to_dict(),
            "event_types_by_step": event_type_by_step.to_dict(),
            "peak_activity_step": step_activity.idxmax()
            if not step_activity.empty
            else None,
            "average_events_per_step": step_activity.mean(),
        }

    def analyze_decision_making_patterns(self) -> dict[str, Any]:
        """Analyze agent decision-making patterns."""
        # Group events by agent and step to identify decision cycles
        decision_cycles = defaultdict(lambda: defaultdict(list))

        for event in self.events:
            agent_id = event.get("agent_id")
            if agent_id is not None and event["event_type"] in [
                "observation",
                "plan",
                "action",
            ]:
                decision_cycles[agent_id][event["step"]].append(event["event_type"])

        # Analyze patterns
        patterns = {}
        for agent_id, steps in decision_cycles.items():
            complete_cycles = 0
            incomplete_cycles = 0
            pattern_types = Counter()

            for _step, events in steps.items():
                event_sequence = tuple(sorted(events))
                pattern_types[event_sequence] += 1

                if "observation" in events and "plan" in events and "action" in events:
                    complete_cycles += 1
                else:
                    incomplete_cycles += 1

            patterns[agent_id] = {
                "complete_decision_cycles": complete_cycles,
                "incomplete_decision_cycles": incomplete_cycles,
                "decision_patterns": dict(pattern_types),
                "total_decision_steps": len(steps),
            }

        return patterns

    def generate_network_graph(self, output_path: str | None = None):
        """Generate communication network visualization."""
        if not self.has_networkx or not self.has_matplotlib:
            return {
                "error": "NetworkX and Matplotlib required for network visualization"
            }

        import matplotlib.pyplot as plt
        import networkx as nx

        communication_data = self.analyze_communication_patterns()
        communication_matrix = communication_data["communication_matrix"]

        # Create directed graph
        graph = nx.DiGraph()

        # Add nodes and edges
        for sender, recipients in communication_matrix.items():
            for recipient, weight in recipients.items():
                graph.add_edge(sender, recipient, weight=weight)

        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, node_color="lightblue", node_size=1000, alpha=0.9
        )

        # Draw edges with thickness proportional to message count
        edges = graph.edges()
        weights = [graph[u][v]["weight"] for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(
            graph,
            pos,
            width=[w / max_weight * 5 for w in weights],
            alpha=0.6,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
        )

        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")

        # Add edge labels with message counts
        edge_labels = {(u, v): str(d["weight"]) for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)

        plt.title("Agent Communication Network", size=16, weight="bold")
        plt.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return {"saved_to": output_path}
        else:
            plt.show()
            return {"displayed": True}

    def generate_activity_timeline(self, output_path: str | None = None):
        """Generate activity timeline visualization."""
        if not self.has_matplotlib:
            return {"error": "Matplotlib required for timeline visualization"}

        import matplotlib.pyplot as plt

        if self.has_seaborn:
            import seaborn as sns

            sns.set_style("whitegrid")

        # Analyze temporal patterns
        temporal_data = self.analyze_temporal_patterns()
        if "error" in temporal_data:
            return temporal_data

        activity_by_step = temporal_data["activity_by_step"]
        steps = sorted(activity_by_step.keys())
        activities = [activity_by_step[step] for step in steps]

        # Create visualization
        plt.figure(figsize=(14, 6))
        plt.plot(steps, activities, marker="o", linewidth=2, markersize=4)
        plt.fill_between(steps, activities, alpha=0.3)

        plt.title("Simulation Activity Timeline", size=16, weight="bold")
        plt.xlabel("Simulation Step", size=12)
        plt.ylabel("Number of Events", size=12)
        plt.grid(True, alpha=0.3)

        # Add peak activity annotation
        peak_step = temporal_data.get("peak_activity_step")
        if peak_step is not None:
            peak_activity = activity_by_step[peak_step]
            plt.annotate(
                f"Peak Activity\nStep {peak_step}: {peak_activity} events",
                xy=(peak_step, peak_activity),
                xytext=(
                    peak_step + len(steps) * 0.1,
                    peak_activity + max(activities) * 0.1,
                ),
                arrowprops={"arrowstyle": "->", "color": "red", "alpha": 0.7},
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
                fontsize=10,
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return {"saved_to": output_path}
        else:
            plt.show()
            return {"displayed": True}

    def export_analysis_report(self, output_path: str | None = None) -> dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = AnalysisReport()

        # Basic statistics
        report.basic_stats = self.get_basic_stats()

        # Agent activity analysis
        report.agent_activity = self.analyze_agent_activity()

        # Communication analysis
        report.communication_patterns = self.analyze_communication_patterns()

        # Temporal analysis
        report.temporal_patterns = self.analyze_temporal_patterns()

        # Decision-making analysis
        report.decision_patterns = self.analyze_decision_making_patterns()

        # Export to JSON
        report_data = {
            "metadata": {
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "source_recording": str(self.recording_path),
                "analyzer_capabilities": {
                    "pandas": self.has_pandas,
                    "networkx": self.has_networkx,
                    "matplotlib": self.has_matplotlib,
                    "seaborn": self.has_seaborn,
                },
            },
            "basic_stats": report.basic_stats,
            "agent_activity": report.agent_activity,
            "communication_patterns": report.communication_patterns,
            "temporal_patterns": report.temporal_patterns,
            "decision_patterns": report.decision_patterns,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        return report_data


class AnalysisReport:
    """Container for analysis results."""

    def __init__(self):
        self.basic_stats = {}
        self.agent_activity = {}
        self.communication_patterns = {}
        self.temporal_patterns = {}
        self.decision_patterns = {}


def load_and_analyze_simulation(recording_path: str) -> SimulationAnalyzer:
    """Convenience function to load and analyze a simulation."""
    return SimulationAnalyzer(recording_path)


def compare_simulations(recording_paths: list[str]) -> dict[str, Any]:
    """Compare multiple simulation recordings."""
    analyzers = [SimulationAnalyzer(path) for path in recording_paths]

    comparison = {
        "simulations": len(analyzers),
        "basic_stats_comparison": {},
        "agent_activity_comparison": {},
        "communication_comparison": {},
    }

    # Compare basic stats
    for i, analyzer in enumerate(analyzers):
        stats = analyzer.get_basic_stats()
        comparison["basic_stats_comparison"][f"simulation_{i}"] = stats

    # Compare agent activities
    for i, analyzer in enumerate(analyzers):
        activity = analyzer.analyze_agent_activity()
        comparison["agent_activity_comparison"][f"simulation_{i}"] = activity

    # Compare communication patterns
    for i, analyzer in enumerate(analyzers):
        comm = analyzer.analyze_communication_patterns()
        comparison["communication_comparison"][f"simulation_{i}"] = comm

    return comparison
