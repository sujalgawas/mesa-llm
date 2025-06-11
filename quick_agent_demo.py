#!/usr/bin/env python3
"""
Quick Agent Demo for Mesa-LLM

This script shows specific agent views without requiring interactive input.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Quick demo of agent viewing capabilities."""

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

        print("\n🎯 Quick Agent Demo")
        print(
            f"Simulation: {viewer.metadata['simulation_id']} | {len(viewer.events)} events | {len(viewer.agent_ids)} agents"
        )

        # Show agent list
        print("\n" + "=" * 60)
        print("📋 ALL AGENTS")
        print("=" * 60)
        viewer.list_agents()

        if viewer.agent_ids:
            # Focus on first agent
            agent_id = viewer.agent_ids[0]

            print("\n" + "=" * 60)
            print(f"🔍 FOCUSING ON AGENT {agent_id}")
            print("=" * 60)

            # Agent summary
            print(f"\n📊 Agent {agent_id} Summary:")
            print("-" * 30)
            viewer.view_agent_summary(agent_id)

            # Agent conversations (if any)
            print(f"\n💬 Agent {agent_id} Conversations:")
            print("-" * 30)
            viewer.view_agent_conversations(agent_id)

            # Agent timeline (first 3 events only for demo)
            print(f"\n📅 Agent {agent_id} Timeline (sample):")
            print("-" * 30)
            agent_events = viewer.agent_events[agent_id][:3]  # Just first 3 events

            for i, event in enumerate(agent_events, 1):
                timestamp_str = event.timestamp.strftime("%H:%M:%S")
                print(
                    f"\n[{i}] Step {event.step} | {timestamp_str} | {event.event_type.upper()}"
                )
                print("   " + event.formatted_content.replace("\n", "\n   "))

        print("\n" + "=" * 60)
        print("✨ INTERACTIVE FEATURES AVAILABLE")
        print("=" * 60)
        print("🚀 For full interactive exploration, run:")
        print("   python view_agents.py")
        print("\n📚 Available commands:")
        print("   • list - Show all agents")
        print("   • summary <id> - Agent statistics")
        print("   • timeline <id> - Complete agent timeline")
        print("   • conversations <id> - All agent messages")
        print("   • decisions <id> - Decision-making process")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
