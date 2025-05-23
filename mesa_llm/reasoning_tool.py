from abc import ABC, abstractmethod, Callable
from dataclasses import dataclass
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.tools.tool_manager import ToolManager
from mesa_llm.memory import Memory, MemoryEntry
from collections import deque
from typing import Any
import json


@dataclass
class Observation:
    """
    Snapshot of everything the agent can see in this step.

    Attributes:
        step (int): The current simulation time step when the observation is made.
        
        self_state (dict): A dictionary containing comprehensive information about the observing agent itself.
            This includes:
            - System prompt or role-specific context for LLM reasoning (if used)
            - Internal state such as morale, fear, aggression, fatigue, etc (behavioural).
            - Agent's current location or spatial coordinates
            - Any other agent-specific metadata that could influence decision-making

        local_state (dict): A dictionary summarizing the state of nearby agents (within the vision radius).
            - A dictionary of neighboring agents, where each key is the "angent's class name + id" and the value is a dictionary containing the following:
            - position of neighbors
            - Internal state or attributes of neighboring agents

    """
    step: int
    self_state: dict           
    local_state: dict          


@dataclass
class Plan:
    """LLM-generated plan that can span ≥1 steps."""
    step: int                # step when the plan was generated
    llm_plan: str             # thoughts and action in sequence
    ttl: int = 1             # steps until re‑planning (ReWOO sets >1)

#subject to change
@dataclass
class Discussion:
    """Discussion between agents."""
    step: int                # step when the discussion was generated
    other_agent_id: int 
    discussion: str           # content of the discussion
               

def _format_observation(observation: Observation) -> str:
    lines = [
        f"Step: {observation.step}",
        "\n[Self State]",
    ]
    for k, v in observation.self_state.items():
        lines.append(f"- {k}: {v}")

    lines.append("\n[Local State of Nearby Agents]")
    for agent_id, agent_info in observation.local_state.items():
        lines.append(f"- {agent_id}:")
        for k, v in agent_info.items():
            lines.append(f"    - {k}: {v}")
    
    return "\n".join(lines)

def _format_plan(plan: Plan) -> str:
    return (
        f"Plan generated at step {plan.step} (valid for {plan.ttl} step(s)):\n"
        f"[Executed Plan]\n{plan.llm_plan.strip()}\n"
    )

def _format_discussion(discussion: Discussion) -> str:
    """Format the discussion once the structure of the datacalss is finalized."""
    pass

def _format_short_term_memory(memory: deque[MemoryEntry]) ->str: 
    if not memory:
        return "No recent memory."

    lines = ["[Short-Term Memory]"]
    for entry in memory:
        lines.append(f"\n[{entry.type.title()} @ Step {entry.step}]")
        lines.append(entry.content.strip()) 
    return "\n".join(lines)


class Reasoning(ABC):
    @abstractmethod
    def plan(
        self,
        prompt: str,
        obs: Observation,
        memory: Memory,
        llm: ModuleLLM,
        tool_manager:ToolManager | None,
        step:int,
        ttl: int=1
    ) -> Plan:
        pass

class ReActReasoning(Reasoning):
    def plan(self, prompt, obs, memory, llm, tool_manager, step):
        long_term_memory = memory.long_term_memory()
        short_term_memory = memory.short_term_memory()
        short_term_memory=_format_short_term_memory(short_term_memory)
        obs_str = _format_observation(obs) #I am going to pass the latest obs seprately from memory so that the llm can pay more attention to it.
        short_term_memory = _format_short_term_memory(memory)
        memory.add_to_memory(type="Observation", content=obs_str, step=step)
        

        system_prompt = f"""
        You are an autonomous agent in a simulation environment.
        You can think about your situation and take actions.
        Use your short-term and long-term memory to guide your behavior.
        You should also use the current observation you have made of the environrment to take suitable actions.
        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {obs_str}

        ---

        # Instructions
        Based on your memory and current situation:
        1. Think through what is happening.
        2. Decide what you should do next.
        3. Respond in the format below:

        Thought: [Explain your reasoning based on memory and current state]  
        Action: [Choose a single action to take]

        Even if multiple actions need to be taken, come up with the first action that needs to be taken at this moment.
        Refer the tools available to you while deciding the action.
        ---

        # Response:
        Thought:
        Action:
        """

        llm.set_system_prompt(system_prompt)
        rsp=llm.genereate(prompt=prompt, tool_schema=tool_manager.get_schema())
        
        response_message = rsp.choices[0].message
        react_plan=Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=_format_plan(react_plan), step=step)
        tool_call_resp=tool_call(rsp, tool_manager)
        memory.add_to_memory(type="Tool_Call_Responses", content=tool_call_resp, step=step)

        return react_plan
    
class CoTReasoning(Reasoning):
    def plan(self, prompt, obs, memory, llm, tool_manager, step):
        long_term_memory = memory.long_term_memory()
        short_term_memory = _format_short_term_memory(memory.short_term_memory())
        obs_str = _format_observation(obs)

        # Add current observation to memory (for record)
        memory.add_to_memory(type="Observation", content=obs_str, step=step)

        system_prompt = f"""
        You are an autonomous agent operating in a simulation.
        Use a detailed step-by-step reasoning process (Chain-of-Thought) to decide your next action.
        Your memory contains information from past experiences, and your observation provides the current context.

        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {obs_str}

        ---

        # Instructions
        Think in multiple reasoning steps before you act.
        Use the format below to respond:

        Thought 1: [Initial reasoning based on the observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [The single best action to take now]

        Keep the reasoning grounded in the current context and relevant history.
        Refer the tools available to you while deciding the action.

        ---

        # Response:
        Thought 1:
        Thought 2:
        Thought 3:
        Thought 4:
        Action:
        """

        llm.set_system_prompt(system_prompt)
        rsp=llm.genereate(prompt=prompt, tool_schema=tool_manager.get_schema())

        response_message = rsp.choices[0].message
        cot_plan = Plan(step=step, llm_plan=response_message, ttl=1)
        memory.add_to_memory(type="Plan", content=_format_plan(cot_plan), step=step)
        tool_call_resp=tool_call(rsp, tool_manager)
        memory.add_to_memory(type="Tool_Call_Responses", content=tool_call_resp, step=step)

        return cot_plan

class ReWOOReasoning(Reasoning):
    def plan(self, prompt, obs, memory, llm, tool_manager, step, ttl):
        long_term_memory = memory.long_term_memory()
        short_term_memory = _format_short_term_memory(memory.short_term_memory())
        obs_str = _format_observation(obs)
        
        # Add current observation to memory
        memory.add_to_memory(type="Observation", content=obs_str, step=step)
        
        system_prompt = f"""
        You are an autonomous agent that creates multi-step plans without re-observing during execution.
        Using the ReWOO (Reasoning WithOut Observation) approach, you will create a comprehensive plan 
        that anticipates multiple steps ahead based on your current observation and memory.
        
        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {obs_str}

        ---

        # Instructions
        Create a detailed multi-step plan that can be executed without needing new observations.
        
        Use this format:

        Plan: [Describe your overall strategy and reasoning]
        
        Step 1: [First action with expected outcome]
        Step 2: [Second action 3ing on Step 1]
        Step 3: [Third action if needed]
        Step 4: [Fourth action if needed]
        Step 5: [Final action if needed]
        
        Contingency: [What to do if things don't go as expected]

        The plan should be comprehensive enough to execute for multiple simulation steps 
        without requiring new environmental observations.
        Refer to available tools when planning actions.

        ---

        # Response:
        Plan:
        Step 1:
        Step 2:
        Step 3:
        Step 4:
        Step 5:
        """

        llm.set_system_prompt(system_prompt)
        rsp=llm.genereate(prompt=prompt, tool_schema=tool_manager.get_schema())
        response_message = rsp.choices[0].message
        rewoo_plan = Plan(step=step, llm_plan=response_message, ttl=ttl) 
        tool_call_resp=tool_call(rsp, tool_manager)
        memory.add_to_memory(type="Tool_Call_Responses", content=tool_call_resp, step=step)

        return rewoo_plan


def tool_call(
    llm_response: Any,
    tool_manager: 'ToolManager'
) -> list[dict]:
    """
    Calls the tools, recommended by the LLM. If the tool has an output it returns the name of the tool and the output else, it returns the name 
    and output as successfully executed
    """
    
    try:
        # Extract response message and tool calls
        response_message = llm_response.choices[0].message
        tool_calls = response_message.tool_calls
        
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
                
                print(f"Processing tool call {i+1}: {function_name}")
                
                # Validate function exists in tool_manager
                if function_name not in tool_manager.tools:
                    raise ValueError(f"Function '{function_name}' not found in ToolManager")
                
                # Parse function arguments
                try:
                    function_args = json.loads(function_args_str)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"Invalid JSON in function arguments: {e}")
                
                # Get the actual function to call from tool_manager
                function_to_call = tool_manager.tools[function_name]
                
                # Call the function with unpacked arguments
                try:
                    function_response = function_to_call(**function_args)
                except TypeError as e:
                    # Handle case where function arguments don't match function signature
                    print(f"Warning: Function call failed with TypeError: {e}")
                    print(f"Attempting to call with filtered arguments...")
                    
                    # Try to filter arguments to match function signature
                    import inspect
                    sig = inspect.signature(function_to_call)
                    filtered_args = {k: v for k, v in function_args.items() if k in sig.parameters}
                    function_response = function_to_call(**filtered_args)
                if not function_response:
                    function_response=f"{function_name} executed successfully"
                
                # Create tool result message
                tool_result = {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "response": str(function_response)
                }
                
                tool_results.append(tool_result) 

            except Exception as e:
                # Handle individual tool call errors
                error_message = f"Error executing tool call {i+1} ({function_name}): {str(e)}"
                print(error_message)
                
                # Create error response
                error_result = {
                    "tool_call_id": tool_call.id,
                    "role": "tool", 
                    "name": function_name,
                    "response": f"Error: {str(e)}"
                }
                
                tool_results.append(error_result)
        return tool_results
        
    except AttributeError as e:
        print(f"Error accessing LLM response structure: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in tool_call: {e}")
        return []




 




