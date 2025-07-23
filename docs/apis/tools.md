# Tools System

The tools system in Mesa-LLM enables agents to interact with their environment and other agents through a structured function-calling interface. Tools represent the concrete actions agents can perform, from basic movement to complex domain-specific behaviors, and are automatically integrated with LLM reasoning through JSON schemas. The tools module provides decorators, managers, and built-in functionality for creating LLM-callable agent actions.

### @tool decorator
**tool(fn=None, \*, tool_manager=None, ignore_agent=True)** → *Callable*

Converts Python functions into LLM-compatible tools by automatically generating JSON schemas from type hints and docstrings. Handles parameter validation, type conversion, and integration with the global tool registry. This module automatically extracts parameter descriptions from Google-style docstrings, injects calling agents into functions expecting an `agent` parameter, and integrates with the global tool registry for automatic availability across all ToolManager instances.

### class ToolManager(extra_tools : list = None)
Manager for registering, organizing, and executing LLM-callable tools with per-agent customization. Supports both global tool registration and per-agent tool customization while maintaining a central registry.

**Attributes:**
- **tools** (dict) - Mapping of tool names to functions
- **instances** (class-level list) - All ToolManager instances for global tool distribution

**Methods:**
- **register(fn)** - Register tool function to this manager
- **add_tool_to_all(fn)** - Add tool to all ToolManager instances
- **get_all_tools_schema(selected_tools=None)** → *list[dict]* - Get OpenAI-compatible schemas
- **call_tools(agent, llm_response)** → *list[dict]* - Execute LLM-recommended tools
- **has_tool(name)** → *bool* - Check if tool is registered

**Tool Execution Flow:**
1. **Tool Registration**: Functions decorated with `@tool` are automatically registered in the global registry
2. **Schema Generation**: Tool decorators analyze function signatures and docstrings to create function calling schemas
3. **LLM Integration**: Reasoning strategies receive tool schemas and can request specific tool calls
4. **Argument Validation**: ToolManager validates LLM-provided arguments against function signatures with automatic type coercion
5. **Execution**: Tools are called with validated arguments, including automatic agent parameter injection
6. **Result Handling**: Tool outputs are captured and added to agent memory for future reasoning

## Built-in Tools

**`move_one_step(agent, direction)`** → *str*
Moves agents one step in specified cardinal/diagonal directions (North, South, East, West, NorthEast, NorthWest, SouthEast, SouthWest). Automatically handles different Mesa grid types including SingleGrid, MultiGrid, OrthogonalGrids, and ContinuousSpace.

**`teleport_to_location(agent, target_coordinates)`** → *str*
Instantly moves agents to specific [x, y] coordinates within grid boundaries. Useful for rapid repositioning or spawning mechanics. Validates coordinates are within environment bounds.

**`speak_to(agent, listener_agents_unique_ids, message)`** → *str*
Enables agent-to-agent communication by sending messages to specified recipients. Messages are automatically added to recipients' memory systems for future reasoning context. Supports both single agent and multiple agent communication.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager

# Creating custom tools
@tool
def change_state(agent: "LLMAgent", new_state: str) -> str:
    """
    Change the agent's internal state.

    Args:
        agent: The agent whose state to change (provided automatically)
        new_state: The new state value to set

    Returns:
        Confirmation message of the state change
    """
    agent.internal_state.append(f"State changed to: {new_state}")
    return f"Agent {agent.unique_id} state changed to {new_state}"

@tool
def arrest_citizen(agent: "LLMAgent", target_agent_id: int) -> str:
    """
    Arrest a citizen agent if they are active and nearby.

    Args:
        agent: The arresting agent (provided automatically)
        target_agent_id: Unique ID of the citizen to arrest

    Returns:
        Result of the arrest attempt
    """
    target = next((a for a in agent.model.agents if a.unique_id == target_agent_id), None)
    if target and target.state == CitizenState.ACTIVE:
        target.jail_sentence_left = 2.0
        target.state = CitizenState.ARRESTED
        return f"Citizen {target_agent_id} has been arrested"
    return f"Could not arrest citizen {target_agent_id}"

# Agent-specific tool configuration
citizen_tool_manager = ToolManager()
cop_tool_manager = ToolManager()

class Citizen(LLMAgent):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.tool_manager = citizen_tool_manager  # Limited tool access

class Cop(LLMAgent):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.tool_manager = cop_tool_manager  # Full tool access

# Tool selection in reasoning
def step(self):
    obs = self.generate_obs()
    plan = self.reasoning.plan(
        obs=obs,
        selected_tools=["move_one_step", "change_state"]  # Restrict available tools for LLM calling
    )
    self.apply_plan(plan)
```