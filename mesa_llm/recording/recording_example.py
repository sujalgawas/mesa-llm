"""
Example demonstrating comprehensive simulation recording in mesa-llm.

This example shows:
1. How to set up recording for a simulation
2. Running a simulation with recording enabled
3. Analyzing the recorded data
4. Generating visualizations and reports
"""

import os
from pathlib import Path

# Import your existing simulation components
from examples.negotiation.model import NegotiationModel
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.recording.analysis import load_and_analyze_simulation
from mesa_llm.recording.integration_hooks import setup_recording_for_existing_simulation


def run_recorded_simulation():
    """Run a simulation with comprehensive recording enabled."""

    # Setup simulation parameters
    api_key = os.getenv("OPENAI_API_KEY")  # or your preferred API key

    # Create model with recording enabled
    model = NegotiationModel(
        initial_buyers=3,
        width=10,
        height=10,
        api_key=api_key,
        reasoning=CoTReasoning,
        llm_model="openai/gpt-4o-mini",
        vision=2,
        seed=42,
    )

    # Set up comprehensive recording
    recording_config = {
        "output_dir": "simulation_recordings",
        "record_observations": True,
        "record_plans": True,
        "record_messages": True,
        "record_actions": True,
        "record_state_changes": True,
        "auto_save_interval": 25,  # Save partial data every 25 events
    }

    recorder = setup_recording_for_existing_simulation(model, recording_config)

    print(f"🎬 Starting recorded simulation with ID: {recorder.simulation_id}")
    print(f"📁 Recordings will be saved to: {recorder.output_dir}")

    # Run simulation for specified number of steps
    num_steps = 20

    for step in range(num_steps):
        print(
            f"📊 Step {step + 1}/{num_steps} - Events recorded: {len(recorder.events)}"
        )
        model.step()

        # Optional: Print some real-time statistics
        if (step + 1) % 5 == 0:
            print(
                f"   📈 Communication events: {len(recorder.get_events_by_type('message'))}"
            )
            print(f"   🧠 Planning events: {len(recorder.get_events_by_type('plan'))}")
            print(
                f"   👁️ Observation events: {len(recorder.get_events_by_type('observation'))}"
            )

    # Save final recording
    recording_path = recorder.save_simulation()
    print(f"\n💾 Simulation recording saved to: {recording_path}")

    # Also export agent memory snapshots
    memory_snapshots = recorder.export_all_agent_memories()
    memory_path = (
        recorder.output_dir / f"memory_snapshots_{recorder.simulation_id}.json"
    )

    import json

    with open(memory_path, "w") as f:
        json.dump(memory_snapshots, f, indent=2, default=str)

    print(f"🧠 Agent memory snapshots saved to: {memory_path}")

    return recording_path


def analyze_simulation_recording(recording_path):
    """Analyze a recorded simulation and generate comprehensive reports."""

    print(f"\n🔍 Analyzing simulation recording: {recording_path}")

    # Load and create analyzer
    analyzer = load_and_analyze_simulation(str(recording_path))

    # Generate summary report
    print("\n📊 Generating analysis report...")
    report = analyzer.generate_summary_report()

    print(f"Simulation ID: {report.simulation_id}")
    print(f"Total Steps: {report.total_steps}")
    print(f"Total Events: {report.total_events}")
    print(f"Agent Count: {report.agent_count}")
    print(f"Event Breakdown: {report.event_statistics}")

    # Communication analysis
    comm_stats = report.communication_stats
    print("\n💬 Communication Analysis:")
    print(f"  Total Messages: {comm_stats['total_messages']}")
    print(f"  Unique Senders: {comm_stats['unique_senders']}")
    print(f"  Avg Messages/Step: {comm_stats['avg_messages_per_step']:.2f}")

    # Agent-specific analysis
    print("\n🤖 Individual Agent Analysis:")
    for agent_id, activity in report.agent_activity.items():
        print(f"  Agent {agent_id}:")
        print(f"    Total Events: {activity['total_events']}")
        print(f"    Event Types: {activity['event_breakdown']}")
        print(
            f"    Active Steps: {activity['activity_span'][0]} - {activity['activity_span'][1]}"
        )

        # Detailed decision-making analysis
        decision_analysis = analyzer.analyze_agent_decision_making(agent_id)
        print(f"    Decisions Made: {decision_analysis['total_decisions']}")
        print(f"    Observations: {decision_analysis['total_observations']}")
        print(f"    Actions: {decision_analysis['total_actions']}")

        reasoning_patterns = decision_analysis["reasoning_patterns"]
        if reasoning_patterns:
            print(
                f"    Tool Usage: {reasoning_patterns.get('tool_usage_frequency', {})}"
            )
            print(
                f"    Reasoning Complexity: {reasoning_patterns.get('reasoning_complexity', 0):.1f} words avg"
            )

    # Generate visualizations
    print("\n📈 Generating visualizations...")

    output_dir = Path("analysis_outputs")
    output_dir.mkdir(exist_ok=True)

    # Communication network visualization
    try:
        analyzer.visualize_communication_network(
            save_path=str(
                output_dir / f"communication_network_{report.simulation_id}.png"
            )
        )
        print("  ✅ Communication network saved")
    except Exception as e:
        print(f"  ⚠️ Communication network visualization failed: {e}")

    # Temporal activity visualization
    try:
        analyzer.visualize_temporal_activity(
            save_path=str(output_dir / f"temporal_activity_{report.simulation_id}.png")
        )
        print("  ✅ Temporal activity chart saved")
    except Exception as e:
        print(f"  ⚠️ Temporal activity visualization failed: {e}")

    # Agent activity heatmap
    try:
        analyzer.visualize_agent_activity_heatmap(
            save_path=str(output_dir / f"activity_heatmap_{report.simulation_id}.png")
        )
        print("  ✅ Activity heatmap saved")
    except Exception as e:
        print(f"  ⚠️ Activity heatmap visualization failed: {e}")

    # Export comprehensive analysis report
    analysis_report_path = output_dir / f"analysis_report_{report.simulation_id}.json"
    analyzer.export_analysis_report(str(analysis_report_path))
    print(f"  ✅ Comprehensive analysis report saved to: {analysis_report_path}")

    return analyzer


def compare_simulation_runs(recording_paths):
    """Compare multiple simulation recordings for pattern analysis."""

    print(f"\n🔄 Comparing {len(recording_paths)} simulation runs...")

    analyzers = [load_and_analyze_simulation(path) for path in recording_paths]
    reports = [analyzer.generate_summary_report() for analyzer in analyzers]

    # Comparison metrics
    comparison = {
        "total_events": [r.total_events for r in reports],
        "communication_counts": [
            r.communication_stats["total_messages"] for r in reports
        ],
        "agent_activity_variance": [],
        "temporal_patterns": [],
    }

    print("📊 Cross-Run Comparison:")
    for i, report in enumerate(reports):
        print(f"  Run {i + 1} ({report.simulation_id}):")
        print(f"    Events: {report.total_events}")
        print(f"    Messages: {report.communication_stats['total_messages']}")
        print(
            f"    Peak Activity Step: {report.temporal_patterns.get('peak_activity_step', 'N/A')}"
        )

    # Calculate averages and variance
    avg_events = sum(comparison["total_events"]) / len(comparison["total_events"])
    avg_messages = sum(comparison["communication_counts"]) / len(
        comparison["communication_counts"]
    )

    print("\n📈 Aggregate Statistics:")
    print(f"  Average Events per Run: {avg_events:.1f}")
    print(f"  Average Messages per Run: {avg_messages:.1f}")

    return comparison


def main():
    """Main function demonstrating the complete recording and analysis workflow."""

    print("🚀 Mesa-LLM Simulation Recording & Analysis Demo")
    print("=" * 50)

    # Step 1: Run recorded simulation
    print("\n🎬 Phase 1: Running Recorded Simulation")
    try:
        recording_path = run_recorded_simulation()
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return

    # Step 2: Analyze the recording
    print("\n🔍 Phase 2: Analyzing Recording")
    try:
        analyzer = analyze_simulation_recording(recording_path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return

    # Step 3: Demonstrate advanced features
    print("\n⚡ Phase 3: Advanced Analysis Features")

    # Query specific events
    message_events = analyzer.get_events_by_type("message")
    plan_events = analyzer.get_events_by_type("plan")

    print("📝 Event Querying:")
    print(f"  Found {len(message_events)} message events")
    print(f"  Found {len(plan_events)} planning events")

    # Show timeline for first agent
    if analyzer.events_df["agent_id"].notna().any():
        first_agent = analyzer.events_df["agent_id"].dropna().iloc[0]
        agent_events = analyzer.get_events_by_agent(int(first_agent))
        print(f"  Agent {int(first_agent)} generated {len(agent_events)} events")

    print("\n✅ Demo completed successfully!")
    print("📁 All outputs saved to: analysis_outputs/")
    print(f"🎥 Recording available at: {recording_path}")


if __name__ == "__main__":
    # Check if we have required dependencies
    try:
        import matplotlib  # noqa: F401
        import networkx  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Install with: pip install matplotlib networkx pandas seaborn")
        exit(1)

    main()
