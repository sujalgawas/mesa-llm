from __future__ import annotations

import inspect
import json
import re
import textwrap
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent
    from mesa_llm.tools.tool_manager import ToolManager


_GLOBAL_TOOL_REGISTRY: dict[str, Callable] = {}
_TOOL_CALLBACKS: list[Callable[[Callable], None]] = []


def add_tool_callback(callback: Callable[[Callable], None]):
    """Add a callback to be called when a new tool is registered"""
    _TOOL_CALLBACKS.append(callback)


# ---------- helper functions ----------------------------------------------------
class DocstringParsingError(Exception):
    """Raised when a Google-style docstring cannot be parsed."""


_ARG_HEADER_RE = re.compile(r"^\s*Args?:\s*$", re.IGNORECASE)
_RET_HEADER_RE = re.compile(r"^\s*Returns?:\s*$", re.IGNORECASE)
_PARAM_LINE_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


def _python_to_json_type(py_type: Any) -> dict[str, Any]:
    # Handle string annotations by converting to actual types
    if isinstance(py_type, str):
        base_type = py_type.split("[")[
            0
        ]  # Extract base type from generics like "tuple[int, int]"
        py_type = (
            getattr(__builtins__, base_type, py_type)
            if hasattr(__builtins__, base_type)
            else py_type
        )

    # Get args before getting origin for better type inference
    args = get_args(py_type)

    # Extract origin type from generics (e.g., tuple from tuple[int, int])
    py_type = get_origin(py_type) or py_type

    # Direct mapping to JSON schema types
    if py_type in (list, tuple):
        # For arrays, we need to specify the items type
        item_type = "integer"  # Default to integer for coordinates
        if args:
            # Try to infer item type from generic args (e.g., tuple[int, int] -> int)
            if args[0] is int:
                item_type = "integer"
            elif args[0] is float:
                item_type = "number"
            elif args[0] is str:
                item_type = "string"
        return {"type": "array", "items": {"type": item_type}}

    type_map = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        bytes: {"type": "string"},
    }
    return type_map.get(py_type, {"type": "object"})


def _parse_docstring(
    func: callable,
    ignore_agent: bool = True,
) -> tuple[str, dict[str, str], str | None]:
    """
    Parse a function's Google-style docstring.
    Args:
        func: The function to parse the docstring of.

    Returns:
        summary: One-line/high-level description that appears before the *Args* section.
        param_desc: Mapping *param name → description* (text only, no types).
        return_desc: Description of the value in the *Returns* section, or *None* if that section is absent.
    """
    # ---------- fetch & pre-process -------------------------------------------------
    raw = inspect.getdoc(func) or ""
    if not raw:
        raise DocstringParsingError(f"{func.__name__} has no docstring.")

    # Normalise indentation & line endings
    lines = textwrap.dedent(raw).strip().splitlines()

    # ---------- locate block boundaries -------------------------------------------
    try:
        args_idx = next(i for i, ln in enumerate(lines) if _ARG_HEADER_RE.match(ln))
    except StopIteration:
        args_idx = None

    try:
        ret_idx = next(i for i, ln in enumerate(lines) if _RET_HEADER_RE.match(ln))
    except StopIteration:
        ret_idx = None

    # Short description = from top up to first blank line or Args:
    cut = (
        args_idx
        if args_idx is not None
        else ret_idx
        if ret_idx is not None
        else len(lines)
    )
    for i, ln in enumerate(lines[:cut]):
        if ln.strip() == "":
            cut = i
            break
    summary = " ".join(ln.strip() for ln in lines[:cut]).strip()

    # ---------- parse *Args* -------------------------------------------------------
    param_desc: dict[str, str] = {}
    if args_idx is not None:
        i = args_idx + 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1  # skip blank lines

        while i < len(lines) and (ret_idx is None or i < ret_idx):
            m = _PARAM_LINE_RE.match(lines[i])
            if not m:
                raise DocstringParsingError(
                    f"Malformed parameter line in {func.__name__}: '{lines[i]}'"
                )
            name, desc = m.groups()
            desc_lines = [desc.rstrip()]
            i += 1
            # grab any following indented continuation lines
            while (
                i < len(lines)
                and (ret_idx is None or i < ret_idx)
                and (lines[i].startswith(" ") or lines[i].startswith("\t"))
                and not _PARAM_LINE_RE.match(
                    lines[i]
                )  # Don't treat other parameters as continuation
            ):
                desc_lines.append(lines[i].strip())
                i += 1
            param_desc[name] = " ".join(desc_lines).strip()
            # skip possible extra blank lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1

    # ---------- parse *Returns* ----------------------------------------------------
    return_desc: str | None = None
    if ret_idx is not None:
        ret_body = [ln.strip() for ln in lines[ret_idx + 1 :] if ln.strip()]
        return_desc = " ".join(ret_body) if ret_body else None

    # ---------- validation ---------------------------------------------------------
    sig_params: list[str] = [
        p.name for p in inspect.signature(func).parameters.values()
    ]
    missing = [p for p in sig_params if p not in param_desc]
    if missing:
        raise DocstringParsingError(
            f"Docstring for {func.__name__} is missing descriptions for: {missing}"
        )

    return summary, param_desc, return_desc


# ---------- decorator ----------------------------------------------------


def tool(
    fn: Callable | None = None,
    *,
    tool_manager: ToolManager | None = None,
    ignore_agent: bool = True,
):
    """
    Decorate a function so it becomes an LLM-callable tool and is auto-registered.

    Args:
        fn: The function to decorate.
        tool_manager : the optional tool manager to add the function to

    Returns:
        The decorated function.
    """

    def decorator(func: Callable):
        name = func.__name__
        description, arg_docs, return_docs = _parse_docstring(func)

        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except NameError:
            # Fallback to using annotations directly if type_hints evaluation fails
            type_hints = getattr(func, "__annotations__", {})

        properties = {}

        # filter out  agent argument if ignore_agent is True
        if ignore_agent:
            required_params = {
                param_name: _param
                for param_name, _param in sig.parameters.items()
                if param_name.lower() != "agent"
            }
        else:
            required_params = sig.parameters

        for param_name, _param in required_params.items():
            raw_type = type_hints.get(param_name, Any)
            type_schema = _python_to_json_type(raw_type)
            properties[param_name] = {
                **type_schema,
                "description": arg_docs.get(param_name, ""),
            }
            if not arg_docs.get(param_name):
                warnings.warn(
                    f'Missing docstring for argument "{param_name}" in tool "{name}"',
                    stacklevel=2,
                )

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description + " returns: " + (return_docs or ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required_params),
                },
            },
        }

        func.__tool_schema__ = schema

        if tool_manager:
            tool_manager.register(func)
        else:
            _GLOBAL_TOOL_REGISTRY[name] = func
            for callback in _TOOL_CALLBACKS:
                callback(func)

        return func

    # If fn is provided, it means @tool was used without parentheses
    if fn is not None:
        return decorator(fn)

    # Otherwise, return the decorator for later use
    return decorator


if __name__ == "__main__":
    # CL to execute this file: python -m mesa_llm.tools.tool_decorator

    @tool
    def dummy_function(agent: LLMAgent, location: tuple[int, int], priority: int = 0):
        """
        Move to a location.
        Args:
            agent: The agent to move
            location: The location to move to.
            priority: The priority of the location.

        Returns:
            The location moved to with priority.
        """
        return f"Moved {agent.unique_id} to {location} with priority {priority}"

    print(json.dumps(dummy_function.__tool_schema__, indent=2))
