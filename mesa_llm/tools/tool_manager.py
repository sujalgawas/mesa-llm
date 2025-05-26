import inspect
import json
from collections.abc import Callable
from typing import Any

from mesa_llm.tools.inbuilt_tools_mock import inbuilt_tools


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

    def get_schema(self) -> list[dict]:
        """Return schema in the liteLLM format"""
        # we need to convert the function signature from python to a JSON schema
        py_to_json_type = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        schema = []
        for name, fn in self.tools.items():
            sig = inspect.signature(fn)
            properties = {}
            required = []

            for param in sig.parameters.values():
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    # skip *args and **kwargs
                    continue
                param_schema = {"description": f"{param.name} parameter"}

                # If type annotation is available
                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation

                    json_type = py_to_json_type.get(annotation)

                    if json_type:
                        param_schema["type"] = json_type
                    else:
                        # fallback: allow any type
                        param_schema["type"] = [
                            "string",
                            "number",
                            "boolean",
                            "object",
                            "array",
                            "null",
                        ]
                else:
                    # No annotation so fallback
                    param_schema["type"] = [
                        "string",
                        "number",
                        "boolean",
                        "object",
                        "array",
                        "null",
                    ]

                properties[param.name] = param_schema

                if param.default == inspect.Parameter.empty:
                    required.append(param.name)

            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": fn.__doc__ or "",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schema  # in case the user wants to change the something like say the parameter description, they will have to get the schema and edit it manually

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool with validated args"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name](**arguments)

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def call_tools(self, llm_response: Any) -> list[dict]:
        """
        Calls the tools, recommended by the LLM. If the tool has an output it returns the name of the tool and the output else, it returns the name
        and output as successfully executed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            A list of tool results of the form:
            [
                {
                    "tool_call_id": str,
                    "role": str,
                    "name": str,
                    "response": str
                },
                ...
            ]
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
                        function_response = function_to_call(**function_args)
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
    print(json.dumps(tool_manager.get_schema(), indent=4))
