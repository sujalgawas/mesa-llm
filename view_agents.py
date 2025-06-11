#!/usr/bin/env python3
"""
Agent Viewer Demo for Mesa-LLM

This script demonstrates the agent-focused viewing capabilities
for recorded mesa-llm simulations.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main function to demonstrate agent viewing."""

    # Check if we have a recording file
    recording_files = list(Path("demo_recordings").glob("*.json"))

    if not recording_files:
        print("❌ No recording files found!")
        print(
            "Please run 'python run_recording_example.py' first to create a recording."
        )
        return

    # Use the most recent recording
    recording_path = max(recording_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using recording: {recording_path}")

    try:
        from mesa_llm.recording.agent_viewer import AgentViewer

        # Create the agent viewer
        viewer = AgentViewer(str(recording_path))

        print("\n🎯 Agent Viewer Demo")
        print(f"Simulation ID: {viewer.metadata['simulation_id']}")
        print(f"Total Events: {len(viewer.events)}")
        print(f"Available Agents: {len(viewer.agent_ids)}")

        # Demo 1: List all agents
        print("\n" + "=" * 50)
        print("📋 Demo 1: Agent List")
        print("=" * 50)
        viewer.list_agents()

        if viewer.agent_ids:
            first_agent = viewer.agent_ids[0]

            # Demo 2: Agent summary
            print("\n" + "=" * 50)
            print(f"📊 Demo 2: Agent {first_agent} Summary")
            print("=" * 50)
            viewer.view_agent_summary(first_agent)

            # Demo 3: Agent timeline
            print("\n" + "=" * 50)
            print(f"📅 Demo 3: Agent {first_agent} Timeline")
            print("=" * 50)
            viewer.view_agent_timeline(first_agent)

            # Demo 4: Agent conversations
            print("\n" + "=" * 50)
            print(f"💬 Demo 4: Agent {first_agent} Conversations")
            print("=" * 50)
            viewer.view_agent_conversations(first_agent)

            # Demo 5: Agent decision-making
            print("\n" + "=" * 50)
            print(f"🧠 Demo 5: Agent {first_agent} Decision-Making")
            print("=" * 50)
            viewer.view_agent_decision_making(first_agent)

        print("\n" + "=" * 50)
        print("🚀 Interactive Mode")
        print("=" * 50)
        print("Starting interactive mode - you can now explore any agent!")
        print(
            "Commands: list, timeline <id>, conversations <id>, decisions <id>, summary <id>, quit"
        )

        # Start interactive mode
        viewer.interactive_mode()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the mesa-llm directory with the recording modules.")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
