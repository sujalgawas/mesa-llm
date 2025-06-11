#!/usr/bin/env python3
"""
Demo script showing mesa-llm simulation recording capabilities.

This script demonstrates the complete recording workflow:
1. Set up and run a mock simulation with recording
2. Analyze the recorded data
3. Generate visualizations and reports
4. View agent-specific data
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Check for required dependencies
missing_deps = []

try:
    import pandas as pd  # noqa: F401
except ImportError:
    missing_deps.append("pandas")

try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:
    missing_deps.append("matplotlib")

try:
    import networkx as nx  # noqa: F401
except ImportError:
    missing_deps.append("networkx")

try:
    import seaborn as sns  # noqa: F401
except ImportError:
    missing_deps.append("seaborn")

if missing_deps:
    print("❌ Missing required dependencies:", ", ".join(missing_deps))
    print("Install with: pip install", " ".join(missing_deps))
    sys.exit(1)

# Mesa-LLM imports
try:
    from mesa_llm.reasoning.reasoning import Observation  # noqa: F401
    from mesa_llm.recording.analysis import SimulationAnalyzer
    from mesa_llm.recording.recorder import SimulationRecorder

    print("✅ Successfully imported recording modules")
except ImportError as e:
    print(f"❌ Failed to import mesa-llm modules: {e}")
    print("Make sure mesa-llm is properly installed")
    sys.exit(1)


def check_dependencies():
    """Check if optional dependencies are available for visualizations."""
    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


def simple_recording_demo():
    """Run a simple recording demonstration without needing API keys."""

    print("🚀 Mesa-LLM Simple Recording Demo")
    print("=" * 40)

    try:
        print("✅ Successfully imported recording modules")

        # Create a mock simulation for demonstration
        class MockModel:
            def __init__(self):
                self.steps = 0
                self.agents = []

            def step(self):
                self.steps += 1

        class MockAgent:
            def __init__(self, unique_id, pos=(0, 0)):
                self.unique_id = unique_id
                self.pos = pos
                self.internal_state = ["active"]

        # Create mock model and agents
        model = MockModel()
        agents = [MockAgent(i, (i, i)) for i in range(3)]
        model.agents = agents

        print(f"📊 Created mock simulation with {len(agents)} agents")

        # Set up recording
        recorder = SimulationRecorder(
            model=model,
            output_dir="demo_recordings",
            record_observations=True,
            record_plans=True,
            record_messages=True,
            record_actions=True,
            record_state_changes=True,
        )

        print(f"🎬 Recording initialized with ID: {recorder.simulation_id}")

        # Simulate some events
        for step in range(5):
            model.step()

            # Record some mock observations
            for agent in agents:
                obs_data = {
                    "step": step,
                    "self_state": {
                        "agent_id": agent.unique_id,
                        "position": agent.pos,
                        "internal_state": agent.internal_state,
                    },
                    "local_state": {
                        f"neighbor_{j}": {"position": (j, j), "state": "active"}
                        for j in range(len(agents))
                        if j != agent.unique_id
                    },
                }
                recorder.record_observation(agent.unique_id, obs_data)

            # Record some mock messages
            if step % 2 == 0:
                recorder.record_message(
                    sender_id=0,
                    message=f"Hello from step {step}!",
                    recipient_ids=[1, 2],
                )

            # Record some mock actions
            for agent in agents:
                action_data = {
                    "action_type": "move",
                    "action_details": {"direction": "north", "distance": 1},
                }
                recorder.record_action(
                    agent_id=agent.unique_id,
                    action_data=action_data,
                )

            print(f"  📈 Step {step + 1}: {len(recorder.events)} events recorded")

        # Save the recording
        recording_path = recorder.save()
        print(f"\n💾 Recording saved to: {recording_path}")

        # Analyze the recording
        print("\n🔍 Analyzing recording...")
        analyzer = SimulationAnalyzer(str(recording_path))
        basic_stats = analyzer.get_basic_stats()

        print("\n📊 Analysis Results:")
        print(f"  Total Events: {basic_stats['total_events']}")
        print(f"  Unique Agents: {basic_stats['unique_agents']}")
        print(f"  Event Types: {basic_stats['event_types']}")
        print(f"  Simulation Steps: {basic_stats['simulation_steps']}")

        # Try to create visualizations if dependencies are available
        if check_dependencies():
            print("\n📈 Creating visualizations...")

            output_dir = Path("demo_analysis")
            output_dir.mkdir(exist_ok=True)

            try:
                result = analyzer.generate_activity_timeline(
                    output_path=str(
                        output_dir / f"activity_{recorder.simulation_id}.png"
                    )
                )
                if "saved_to" in result:
                    print("  ✅ Temporal activity chart saved")
                else:
                    print(f"  ⚠️ Visualization issue: {result}")
            except Exception as e:
                print(f"  ⚠️ Visualization failed: {e}")

            # Export analysis report
            analysis_path = output_dir / f"report_{recorder.simulation_id}.json"
            try:
                analyzer.export_analysis_report(str(analysis_path))
                print(f"  ✅ Analysis report saved to: {analysis_path}")
            except Exception as e:
                print(f"  ⚠️ Analysis report export failed: {e}")
                print("  📊 Basic stats available in memory")

        print("\n✅ Demo completed successfully!")
        print("📁 Outputs saved to: demo_recordings/ and demo_analysis/")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the mesa-llm directory")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def negotiation_example():
    """Try to run the full negotiation example if API key is available."""

    print("\n🤝 Attempting Negotiation Example")
    print("=" * 40)

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        print("⚠️ No API key found in environment variables.")
        print("To run the full negotiation example, set one of:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GEMINI_API_KEY")
        return False

    try:
        from examples.negotiation.model import NegotiationModel
        from mesa_llm.reasoning.cot import CoTReasoning
        from mesa_llm.recording.analysis import load_and_analyze_simulation
        from mesa_llm.recording.integration_hooks import (
            setup_recording_for_existing_simulation,
        )

        print("✅ Imported negotiation model components")

        # Create negotiation model
        model = NegotiationModel(
            initial_buyers=2,
            width=5,
            height=5,
            api_key=api_key,
            reasoning=CoTReasoning,
            llm_model="openai/gpt-4o-mini"
            if "OPENAI" in api_key
            else "gemini/gemini-2.0-flash",
            vision=2,
            seed=42,
        )

        print(f"🏪 Created negotiation model with {len(model.agents)} agents")

        # Set up recording
        recorder = setup_recording_for_existing_simulation(
            model,
            output_dir="negotiation_recordings",
            record_observations=True,
            record_plans=True,
            record_messages=True,
            record_actions=True,
            auto_save_interval=10,
        )

        print(f"🎬 Recording enabled for simulation {recorder.simulation_id}")

        # Run simulation
        num_steps = 5  # Keep it short for demo
        for step in range(num_steps):
            print(f"📊 Step {step + 1}/{num_steps} - Events: {len(recorder.events)}")
            model.step()

        # Save and analyze
        recording_path = recorder.save()
        print(f"💾 Negotiation recording saved to: {recording_path}")

        analyzer = load_and_analyze_simulation(str(recording_path))
        basic_stats = analyzer.get_basic_stats()
        comm_patterns = analyzer.analyze_communication_patterns()

        print("\n📊 Negotiation Analysis:")
        print(f"  Total Events: {basic_stats['total_events']}")
        print(f"  Messages Exchanged: {comm_patterns['total_messages']}")
        print(f"  Active Agents: {basic_stats['unique_agents']}")

        return True

    except ImportError as e:
        print(f"❌ Failed to import negotiation components: {e}")
        return False
    except Exception as e:
        print(f"❌ Negotiation example failed: {e}")
        return False


def main():
    """Main function to run the recording demo."""

    print("🎯 Mesa-LLM Recording System Demo")
    print("This script demonstrates the simulation recording capabilities.")
    print("\n" + "=" * 50)

    # Always run the simple demo
    success = simple_recording_demo()

    if success:
        # Try the negotiation example if API key is available
        negotiation_example()

    print("\n" + "=" * 50)
    print("📚 For more advanced examples, see:")
    print("  - mesa_llm/recording/recording_example.py")
    print("  - Mesa-LLM documentation")


if __name__ == "__main__":
    main()
