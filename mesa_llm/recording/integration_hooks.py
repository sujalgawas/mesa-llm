"""
Integration hooks for adding recording capabilities to existing mesa-llm simulations.

This module provides tools for non-invasive integration of recording capabilities
into existing simulations without requiring major code changes.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from .recorder import SimulationRecorder


class RecordingMixin:
    """
    Mixin class to add recording capabilities to existing classes.

    This can be used with both Model and Agent classes to automatically
    record relevant events.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder: SimulationRecorder | None = None

    def set_recorder(self, recorder: SimulationRecorder):
        """Set the recorder for this instance."""
        self.recorder = recorder

    def record_event(
        self,
        event_type: str,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        """Record an event if recorder is available."""
        if self.recorder:
            agent_id = getattr(self, "unique_id", None)
            self.recorder.record_event(event_type, content, agent_id, metadata)


def record_method_call(event_type: str, content_extractor: Callable | None = None):
    """
    Decorator to automatically record agent method calls.

    Args:
        event_type: Type of event to record
        content_extractor: Function to extract content from method arguments
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Execute the original method
            result = method(self, *args, **kwargs)

            # Record the event if recorder is available
            if hasattr(self, "recorder") and self.recorder:
                if content_extractor:
                    content = content_extractor(self, *args, **kwargs)
                else:
                    content = {
                        "method": method.__name__,
                        "args": args,
                        "kwargs": kwargs,
                        "result": result,
                    }

                agent_id = getattr(self, "unique_id", None)
                self.recorder.record_event(
                    event_type=event_type,
                    content=content,
                    agent_id=agent_id,
                    metadata={"source": f"method_{method.__name__}"},
                )

            return result

        return wrapper

    return decorator


# Example content extractors for common LLM agent methods


def extract_observation_content(agent, observation):
    """Extract content from an observe method call."""
    return {
        "step": observation.step,
        "self_state": observation.self_state,
        "local_state": observation.local_state,
    }


def extract_plan_content(agent, plan):
    """Extract content from a plan method call."""
    plan_content = {}
    if hasattr(plan.llm_plan, "content"):
        plan_content["content"] = plan.llm_plan.content
    if hasattr(plan.llm_plan, "tool_calls") and plan.llm_plan.tool_calls:
        plan_content["tool_calls"] = [
            {"function": call.function.name, "arguments": call.function.arguments}
            for call in plan.llm_plan.tool_calls
        ]

    return {"step": plan.step, "ttl": plan.ttl, "plan_content": plan_content}


def extract_action_content(agent, action_type, action_details, tool_responses=None):
    """Extract content from an action method call."""
    return {
        "action_type": action_type,
        "action_details": action_details,
        "tool_responses": tool_responses or [],
    }


# Example of how to extend LLMAgent with recording
class RecordingLLMAgent:
    """
    Example showing how to extend LLMAgent with recording capabilities.

    This would typically be done by modifying the existing LLMAgent class
    or creating a subclass that users can opt into.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder: SimulationRecorder | None = None

    def set_recorder(self, recorder: SimulationRecorder):
        """Set the recorder for this agent."""
        self.recorder = recorder

    @record_method_call("observation", extract_observation_content)
    def observe(self, observation):
        """Record observations automatically."""
        return super().observe(observation)

    @record_method_call("plan", extract_plan_content)
    def plan(self, plan):
        """Record plans automatically."""
        return super().plan(plan)

    @record_method_call("action", extract_action_content)
    def act(self, action_type, action_details, tool_responses=None):
        """Record actions automatically."""
        return super().act(action_type, action_details, tool_responses)


def add_recorder_to_model(model_class):
    """
    Class decorator to add recorder support to a Model class.

    Usage:
        @add_recorder_to_model
        class MyModel(Model):
            pass
    """

    class RecordingModel(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorder: SimulationRecorder | None = None

        def setup_recording(self, **recorder_kwargs):
            """Set up recording for this model."""
            self.recorder = SimulationRecorder(self, **recorder_kwargs)

            # Add recorder to all agents
            for agent in self.schedule.agents:
                if hasattr(agent, "set_recorder"):
                    agent.set_recorder(self.recorder)

            return self.recorder

        def step(self):
            """Override step to record model events."""
            if self.recorder:
                self.recorder.record_model_event(
                    event_type="step_start", content={"step": self.schedule.steps}
                )

            # Execute normal step
            super().step()

            if self.recorder:
                self.recorder.record_model_event(
                    event_type="step_end", content={"step": self.schedule.steps}
                )

                # Track agent states
                for agent in self.schedule.agents:
                    current_state = {
                        "position": getattr(agent, "pos", None),
                        "internal_state": getattr(agent, "internal_state", []),
                        "is_speaking": getattr(agent, "is_speaking", False),
                    }
                    self.recorder.track_agent_state(agent.unique_id, current_state)

    return RecordingModel


# Runtime integration functions


def add_recording_to_agent(agent, recorder: SimulationRecorder):
    """
    Add recording to an existing agent instance.

    This function shows how to monkey-patch recording onto existing agents.
    """
    agent.recorder = recorder

    # Store original methods
    original_observe = getattr(agent, "observe", None)
    original_plan = getattr(agent, "plan", None)
    original_act = getattr(agent, "act", None)

    # Add recording wrappers
    if original_observe:

        def recorded_observe(observation):
            result = original_observe(observation)
            if agent.recorder:
                content = extract_observation_content(agent, observation)
                agent.recorder.record_observation(agent.unique_id, content)
            return result

        agent.observe = recorded_observe

    if original_plan:

        def recorded_plan(plan):
            result = original_plan(plan)
            if agent.recorder:
                content = extract_plan_content(agent, plan)
                agent.recorder.record_plan(agent.unique_id, content)
            return result

        agent.plan = recorded_plan

    if original_act:

        def recorded_act(action_type, action_details, tool_responses=None):
            result = original_act(action_type, action_details, tool_responses)
            if agent.recorder:
                content = extract_action_content(
                    agent, action_type, action_details, tool_responses
                )
                agent.recorder.record_action(agent.unique_id, content)
            return result

        agent.act = recorded_act


def setup_recording_for_existing_simulation(
    model, **recorder_kwargs
) -> SimulationRecorder:
    """
    Set up recording for an existing model and its agents.

    This is useful for adding recording to simulations that are already running.
    """
    recorder = SimulationRecorder(model, **recorder_kwargs)

    # Add recording to the model
    model.recorder = recorder

    # Add recording to all agents
    for agent in model.schedule.agents:
        add_recording_to_agent(agent, recorder)

    # Monkey-patch the model's step method
    original_step = model.step

    def recorded_step():
        recorder.record_model_event(
            event_type="step_start", content={"step": model.schedule.steps}
        )

        original_step()

        recorder.record_model_event(
            event_type="step_end", content={"step": model.schedule.steps}
        )

        # Track agent states
        for agent in model.schedule.agents:
            current_state = {
                "position": getattr(agent, "pos", None),
                "internal_state": getattr(agent, "internal_state", []),
                "is_speaking": getattr(agent, "is_speaking", False),
            }
            recorder.track_agent_state(agent.unique_id, current_state)

    model.step = recorded_step

    return recorder
