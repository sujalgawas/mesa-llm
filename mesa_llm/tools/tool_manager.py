import inspect
import json
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mesa_llm.tools.inbuilt_tools_mock import inbuilt_tools

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ToolManager:
    """
    The user can use instances of ToolManager to register functions as tools through the decorator.
    The user can also use the ToolManager instance to get the schema of the tools, call a tool with validated arguments, and check if a tool is registered.
    Moreover, the user can group like tools together by creating a new ToolManager instance and registering the tools to it.
    So if agent A requires tools A1, A2, and A3, and agent B requires tools B1, B2, and B3, the user can create two ToolManager instances: tool_manager_A and tool_manager_B.

    Attributes:
        tools: A dictionary of tools of the form {name: function}. E.g. {"get_current_weather": get_current_weather}.
    """

    def __init__(self):
        self.tools: dict[str, Callable] = {}

        for tool in inbuilt_tools:
            self.register(tool)

    def register(self, fn: Callable):
        """Register a tool function by name"""
        name = fn.__name__
        self.tools[name] = fn  # storing the name & function pair as a dictionary

    def get_tool_schema(self, fn, schema_name):
        doc = inspect.getdoc(fn) or ""  # get the docstring of the function
        lines = doc.splitlines()
        args_start = next(
            (i for i, line in enumerate(lines) if line.strip().lower() == "args:"), None
        )  # find the line that contains "Args:"
        desc = (
            " ".join(lines[:args_start]).strip()
            if args_start is not None
            else doc.strip()
        )  # get the description of the function

        # Parse Args
        arg_docs = {}
        if args_start is not None:
            current = None
            for line in lines[
                args_start + 1 :
            ]:  # iterate over the lines after the args section (stop at "Returns:")
                stripped_line = line.strip()
                if not stripped_line or stripped_line.lower().startswith("returns"):
                    break
                if ":" in stripped_line:
                    if current:
                        arg_docs[current[0]] = " ".join(current[1]).strip()
                    current = [
                        stripped_line.split(":", 1)[0].strip(),
                        [stripped_line.split(":", 1)[1].strip()],
                    ]  # split the line into a key and a value
                elif current:
                    current[1].append(stripped_line)
            if current:
                arg_docs[current[0]] = " ".join(current[1]).strip()

        # Build schema
        sig = inspect.signature(fn)
        props = {
            name: {
                "type": "array"
                if str(prm.annotation).startswith(("tuple", "list"))
                else "Any"
                if prm.annotation is inspect._empty
                else prm.annotation
                if isinstance(prm.annotation, str)
                else getattr(prm.annotation, "__name__", str(prm.annotation)),
                "description": arg_docs.get(name, ""),
            }
            for name, prm in sig.parameters.items()
        }

        # Warn for missing descriptions
        for name in sig.parameters:
            if name not in arg_docs:
                warnings.warn(f'Missing description for "{name}"', stacklevel=2)

        return {
            "type": "function",
            "function": {
                "name": schema_name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(sig.parameters),
                },
            },
        }

    def get_all_tools_schema(self) -> list[dict]:
        """Return schema in the liteLLM format"""

        schema = []
        for name, fn in self.tools.items():
            schema.append(self.get_tool_schema(fn, name))

        return schema

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool with validated args"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name](**arguments)

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def call_tools(self, agent: "LLMAgent", llm_response: Any) -> list[dict]:
        """
        Calls the tools, recommended by the LLM. If the tool has an output it returns the name of the tool and the output else, it returns the name
        and output as successfully executed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            A list of tool results
        """

        try:
            # Extract response message and tool calls
            tool_calls = llm_response.tool_calls

            # Check if tool_calls exists and is not None
            if not tool_calls:
                print("No tool calls found in LLM response")
                return []

            print(f"Found {len(tool_calls)} tool call(s)")

            tool_results = []

            # Process each tool call
            for i, tool_call in enumerate(tool_calls):
                try:
                    # Extract function details
                    function_name = tool_call.function.name
                    function_args_str = tool_call.function.arguments
                    tool_call_id = tool_call.id

                    print(f"Processing tool call {i + 1}: {function_name}")

                    # Validate function exists in tool_manager
                    if function_name not in self.tools:
                        raise ValueError(
                            f"Function '{function_name}' not found in ToolManager"
                        )

                    # Parse function arguments
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Invalid JSON in function arguments: {e}"
                        ) from e

                    # Get the actual function to call from tool_manager
                    function_to_call = self.tools[function_name]

                    # Call the function with unpacked arguments
                    try:
                        function_response = function_to_call(
                            agent=agent, **function_args
                        )
                    except TypeError as e:
                        # Handle case where function arguments don't match function signature
                        print(f"Warning: Function call failed with TypeError: {e}")
                        print("Attempting to call with filtered arguments...")

                        # Try to filter arguments to match function signature
                        import inspect

                        sig = inspect.signature(function_to_call)
                        filtered_args = {
                            k: v
                            for k, v in function_args.items()
                            if k in sig.parameters
                        }
                        function_response = function_to_call(**filtered_args)
                    if not function_response:
                        function_response = f"{function_name} executed successfully"

                    # Create tool result message
                    tool_result = {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "response": str(function_response),
                    }

                    tool_results.append(tool_result)

                except Exception as e:
                    # Handle individual tool call errors
                    error_message = (
                        f"Error executing tool call {i + 1} ({function_name}): {e!s}"
                    )
                    print(error_message)

                    # Create error response
                    error_result = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "response": f"Error: {e!s}",
                    }

                    tool_results.append(error_result)
            return tool_results

        except AttributeError as e:
            print(f"Error accessing LLM response structure: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in call_tools: {e}")
            return []


if __name__ == "__main__":
    tool_manager = ToolManager()
    print(json.dumps(tool_manager.get_all_tools_schema(), indent=4))
