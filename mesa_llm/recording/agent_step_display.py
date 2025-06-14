import json
import re

from rich.console import Console
from rich.panel import Panel

console = Console()


def display_agent_step(
    step, agent_class, agent_id, observation=None, plan_content=None, tool_calls=None
):
    """Display agent step with rich formatting - simple and clean."""

    lines = []

    # Observation section
    if observation:
        lines.append("[bold cyan][Observation][/bold cyan] :")

        # Self State
        if "self_state" in observation:
            self_state = observation["self_state"]
            loc = self_state.get("location", "Unknown")
            internal = self_state.get("internal_state", [])
            loc_str = (
                f"[{loc[0]},{loc[1]}]"
                if isinstance(loc, list | tuple) and len(loc) >= 2
                else str(loc)
            )
            lines.append(
                f"   └── [cyan][Self State][/cyan] : location: {loc_str}, internal_state: {internal}"
            )

        # Local State
        if "local_state" in observation:
            local_state = observation["local_state"]
            if local_state:
                agents = []
                for name, info in local_state.items():
                    pos = info.get("position", "Unknown")
                    pos_str = (
                        f"[{pos[0]},{pos[1]}]"
                        if isinstance(pos, list | tuple) and len(pos) >= 2
                        else str(pos)
                    )
                    agents.append(f"{name} (loc : {pos_str})")
                lines.append(f"   └── [cyan][Local State][/cyan] : {', '.join(agents)}")
        lines.append("")

    # Plan section
    if plan_content:
        lines.append("[bold cyan][Plan][/bold cyan]")

        # Parse reasoning and action
        reasoning_match = re.search(
            r"Reasoning:\s*(.+?)(?=\n\s*Action:|$)",
            plan_content,
            re.DOTALL | re.IGNORECASE,
        )
        action_match = re.search(
            r"Action:\s*(.+?)$", plan_content, re.DOTALL | re.IGNORECASE
        )

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            lines.append(f'   └──  [cyan][Reasoning][/cyan] : "{reasoning}"')

        if action_match:
            action = action_match.group(1).strip()
            lines.append(f'   └──  [cyan][Action][/cyan] : "{action}"')

        lines.append("")

    # Tool calls section
    if tool_calls:
        lines.append("[bold cyan][Tool_call][/bold cyan] :")
        for tool_call in tool_calls:
            if "function" in tool_call:
                func = tool_call["function"]
                lines.append(
                    f"   └──  [cyan]\\[tool][/cyan] : {func.get('name', 'unknown')}"
                )
                try:
                    args = json.loads(func.get("arguments", "{}"))
                    for arg_name, arg_value in args.items():
                        lines.append(
                            f"   └──  [cyan]\\[arg][/cyan] ({arg_name}) :  {arg_value}"
                        )
                except Exception as e:
                    print(f"Error parsing tool call arguments: {e}")

        lines.append("")

    # Create and display panel
    content = "\n".join(lines)
    title = f"Step [bold purple]{step}[/bold purple] [bold]|[/bold] {agent_class} [bold purple]{agent_id}[/bold purple]"
    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style="bright_blue",
        padding=(0, 1),
    )
    console.print(panel)


def extract_tool_calls(plan_message):
    """Extract tool calls from plan message."""
    if not hasattr(plan_message, "tool_calls") or not plan_message.tool_calls:
        return None

    tool_calls = []
    for tc in plan_message.tool_calls:
        tool_calls.append(
            {
                "function": {
                    "name": tc.function.name if hasattr(tc, "function") else "",
                    "arguments": tc.function.arguments
                    if hasattr(tc, "function")
                    else "{}",
                }
            }
        )
    return tool_calls
