import json
from unittest.mock import Mock

import pytest

from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import (
    Observation,
    Plan,
    Reasoning,
)
from mesa_llm.reasoning.rewoo import ReWOOReasoning


class TestObservation:
    """Test cases for the Observation dataclass."""

    def test_observation_creation(self):
        """Test basic observation creation."""
        obs = Observation(
            step=1,
            self_state={"location": (2, 3), "health": 100},
            local_state={"agent_1": {"position": (1, 1), "state": "active"}},
        )

        assert obs.step == 1
        assert obs.self_state["location"] == (2, 3)
        assert obs.local_state["agent_1"]["position"] == (1, 1)

    def test_observation_str_representation(self):
        """Test the string representation of observation."""
        obs = Observation(
            step=0,
            self_state={"location": (0, 0), "health": 50},
            local_state={"enemy_1": {"position": (1, 0), "state": "hostile"}},
        )

        str_repr = str(obs)
        # The string representation is now the default dataclass __str__
        assert "step=0" in str_repr
        assert "location" in str_repr
        assert "enemy_1" in str_repr


class TestPlan:
    """Test cases for the Plan dataclass."""

    def test_plan_creation(self):
        """Test basic plan creation."""
        mock_message = Mock()
        mock_message.content = "Move to position (2, 3)"

        plan = Plan(step=1, llm_plan=mock_message, ttl=2)

        assert plan.step == 1
        assert plan.llm_plan == mock_message
        assert plan.ttl == 2

    def test_plan_default_ttl(self):
        """Test plan creation with default TTL."""
        mock_message = Mock()
        plan = Plan(step=1, llm_plan=mock_message)

        assert plan.ttl == 1

    def test_plan_str_representation(self):
        """Test Plan string representation."""
        mock_message = Mock()
        mock_message.content = "Test plan content"

        plan = Plan(step=1, llm_plan=mock_message)
        str_repr = str(plan)
        assert "Test plan content" in str_repr


class TestReActReasoning:
    """Test cases for ReActReasoning class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()
        agent.model = Mock()
        agent.model.steps = 1
        agent.unique_id = "test_agent"
        agent.recorder = None

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Long term memory content"
        agent.memory.format_short_term.return_value = "Short term memory content"
        agent.memory.add_to_memory = Mock()
        agent.memory.short_term_memory = []

        # Mock _step_display_data as a dictionary to avoid assignment errors
        agent._step_display_data = {}

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = [
            {"function": {"name": "test_tool", "description": "Test tool"}}
        ]

        # Setup LLM mock with proper response structure
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = json.dumps(
            {
                "reasoning": "I need to analyze the situation",
                "action": "Move to a strategic position",
            }
        )
        mock_response.choices = [Mock(message=mock_message)]

        # Second response for execute_tool_call
        mock_tool_response = Mock()
        mock_tool_message = Mock()
        mock_tool_message.content = "Tool executed successfully"
        mock_tool_message.tool_calls = []
        mock_tool_response.choices = [Mock(message=mock_tool_message)]

        agent.llm.generate.side_effect = [mock_response, mock_tool_response]
        agent.llm.set_system_prompt = Mock()

        return agent

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation for testing."""
        return Observation(
            step=1,
            self_state={"location": (1, 1), "health": 80},
            local_state={"enemy_1": {"position": (2, 2), "state": "aggressive"}},
        )

    def test_react_reasoning_initialization(self, mock_agent):
        """Test ReActReasoning initialization."""
        reasoning = ReActReasoning(mock_agent)
        assert reasoning.agent == mock_agent

    def test_react_reasoning_plan(self, mock_agent, sample_observation):
        """Test ReActReasoning plan method."""
        reasoning = ReActReasoning(mock_agent)
        prompt = "Analyze the situation and take action"

        plan = reasoning.plan(prompt, sample_observation, ttl=1)

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)
        assert (
            plan.step == mock_agent.model.steps
        )  # execute_tool_call uses agent.model.steps
        assert plan.ttl == 1

        # Verify LLM interactions
        assert mock_agent.llm.set_system_prompt.call_count >= 1
        assert mock_agent.llm.generate.call_count >= 1

        # Verify memory interactions
        assert mock_agent.memory.add_to_memory.call_count >= 1

    def test_react_reasoning_plan_with_selected_tools(
        self, mock_agent, sample_observation
    ):
        """Test ReActReasoning plan method with selected_tools parameter."""
        reasoning = ReActReasoning(mock_agent)
        prompt = "Analyze the situation with specific tools"
        selected_tools = ["tool1", "tool2"]

        plan = reasoning.plan(
            prompt, sample_observation, ttl=1, selected_tools=selected_tools
        )

        # Verify that get_all_tools_schema was called with selected_tools
        # Note: There may be multiple calls due to execute_tool_call, so check if any call used selected_tools
        call_args_list = mock_agent.tool_manager.get_all_tools_schema.call_args_list
        has_selected_call = any(
            args == (selected_tools,) for args, kwargs in call_args_list
        )
        assert has_selected_call, (
            f"Expected call with selected_tools {selected_tools}, got calls: {call_args_list}"
        )

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)
        assert (
            plan.step == mock_agent.model.steps
        )  # execute_tool_call uses agent.model.steps


class TestCoTReasoning:
    """Test cases for CoTReasoning class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()
        agent.model = Mock()
        agent.model.steps = 1
        agent.unique_id = "test_agent"
        agent.recorder = None

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Long term memory content"
        agent.memory.format_short_term.return_value = "Short term memory content"
        agent.memory.add_to_memory = Mock()

        # Mock _step_display_data as a dictionary to avoid assignment errors
        agent._step_display_data = {}

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = [
            {"function": {"name": "test_tool", "description": "Test tool"}}
        ]

        # Setup LLM mock with proper response structure
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = (
            "Thought 1: I see an enemy\nThought 2: I should move\nAction: Move away"
        )
        mock_response.choices = [Mock(message=mock_message)]

        # Second response for execute_tool_call
        mock_tool_response = Mock()
        mock_tool_message = Mock()
        mock_tool_message.content = "Tool executed successfully"
        mock_tool_message.tool_calls = []
        mock_tool_response.choices = [Mock(message=mock_tool_message)]

        agent.llm.generate.side_effect = [mock_response, mock_tool_response]
        agent.llm.set_system_prompt = Mock()

        return agent

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation for testing."""
        return Observation(
            step=2,
            self_state={"location": (3, 3), "health": 60},
            local_state={"ally_1": {"position": (3, 4), "state": "friendly"}},
        )

    def test_cot_reasoning_initialization(self, mock_agent):
        """Test CoTReasoning initialization."""
        reasoning = CoTReasoning(mock_agent)
        assert reasoning.agent == mock_agent

    def test_cot_reasoning_plan(self, mock_agent, sample_observation):
        """Test CoTReasoning plan method."""
        reasoning = CoTReasoning(mock_agent)
        prompt = "Think step by step and decide your action"

        plan = reasoning.plan(prompt, sample_observation, ttl=1)

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)
        assert plan.step == sample_observation.step + 1  # CoT uses obs.step + 1
        assert plan.ttl == 1

        # Verify system prompt contains CoT instructions
        system_prompt_calls = mock_agent.llm.set_system_prompt.call_args_list
        assert len(system_prompt_calls) >= 1
        # Check that at least one call contains chain of thought instructions
        cot_found = any(
            "Chain-of-Thought" in call[0][0] for call in system_prompt_calls
        )
        assert cot_found

        # Verify memory interactions
        assert mock_agent.memory.add_to_memory.call_count >= 1

    def test_cot_reasoning_plan_with_selected_tools(
        self, mock_agent, sample_observation
    ):
        """Test CoTReasoning plan method with selected_tools parameter."""
        reasoning = CoTReasoning(mock_agent)
        prompt = "Think step by step with specific tools"
        selected_tools = ["tool1", "tool2"]

        plan = reasoning.plan(
            prompt, sample_observation, ttl=1, selected_tools=selected_tools
        )

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)
        assert plan.step == sample_observation.step + 1  # CoT uses obs.step + 1


class TestReWOOReasoning:
    """Test cases for ReWOOReasoning class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()
        agent.model = Mock()
        agent.model.steps = 1
        agent.unique_id = "test_agent"
        agent.recorder = None
        agent.generate_obs = Mock()

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Long term memory content"
        agent.memory.format_short_term.return_value = "Short term memory content"
        agent.memory.add_to_memory = Mock()

        # Mock _step_display_data as a dictionary to avoid assignment errors
        agent._step_display_data = {}

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = [
            {"function": {"name": "test_tool", "description": "Test tool"}}
        ]

        # Create mock observation for generate_obs
        mock_obs = Observation(
            step=5,
            self_state={"location": (5, 5), "health": 90},
            local_state={"target_1": {"position": (6, 6), "state": "unknown"}},
        )
        agent.generate_obs.return_value = mock_obs

        # Setup LLM mock with proper response structure
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Multi-step plan created"
        mock_response.choices = [Mock(message=mock_message)]

        # Second response for execute_tool_call
        mock_tool_response = Mock()
        mock_tool_message = Mock()
        mock_tool_message.content = "Tool executed successfully"
        mock_tool_message.tool_calls = []
        mock_tool_response.choices = [Mock(message=mock_tool_message)]

        agent.llm.generate.side_effect = [mock_response, mock_tool_response]
        agent.llm.set_system_prompt = Mock()

        return agent

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation for testing."""
        return Observation(
            step=5,
            self_state={"location": (5, 5), "health": 90},
            local_state={"target_1": {"position": (6, 6), "state": "unknown"}},
        )

    def test_rewoo_reasoning_initialization(self, mock_agent):
        """Test ReWOOReasoning initialization."""
        reasoning = ReWOOReasoning(mock_agent)
        assert reasoning.agent == mock_agent
        assert reasoning.remaining_tool_calls == 0
        assert reasoning.current_plan is None
        assert reasoning.current_obs is None

    def test_rewoo_reasoning_plan(self, mock_agent, sample_observation):
        """Test ReWOOReasoning plan method."""
        reasoning = ReWOOReasoning(mock_agent)
        prompt = "Create a comprehensive multi-step plan"

        plan = reasoning.plan(prompt)

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)
        assert (
            plan.step == mock_agent.model.steps
        )  # ReWOO execute_tool_call uses agent.model.steps
        assert plan.ttl == 1

        # Verify LLM interactions
        assert mock_agent.llm.set_system_prompt.call_count >= 1
        assert mock_agent.llm.generate.call_count >= 1

        # Verify memory interactions
        assert mock_agent.memory.add_to_memory.call_count >= 1

    def test_rewoo_reasoning_plan_with_selected_tools(
        self, mock_agent, sample_observation
    ):
        """Test ReWOOReasoning plan method with selected_tools parameter."""
        reasoning = ReWOOReasoning(mock_agent)
        prompt = "Create a plan with specific tools"
        selected_tools = ["tool1", "tool2"]

        plan = reasoning.plan(prompt, selected_tools=selected_tools)

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)

    def test_rewoo_reasoning_plan_continuation(self, mock_agent, sample_observation):
        """Test ReWOOReasoning plan continuation with remaining tool calls."""
        reasoning = ReWOOReasoning(mock_agent)

        # Setup a plan with tool calls
        mock_plan = Mock()
        mock_plan.tool_calls = [Mock(), Mock(), Mock()]
        reasoning.current_plan = mock_plan
        reasoning.remaining_tool_calls = 2
        reasoning.current_obs = sample_observation

        plan = reasoning.plan("Continue plan")

        # Verify it returns a continuation plan
        assert isinstance(plan, Plan)
        assert reasoning.remaining_tool_calls == 1  # Should decrease by 1


class TestReasoningAbstractClass:
    """Test cases for the abstract Reasoning class."""

    def test_reasoning_is_abstract(self):
        """Test that Reasoning cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Reasoning(Mock())

    def test_reasoning_subclass_must_implement_plan(self):
        """Test that subclasses must implement the plan method."""

        class IncompleteReasoning(Reasoning):
            def __init__(self, agent):
                super().__init__(agent)

        with pytest.raises(TypeError):
            IncompleteReasoning(Mock())

    def test_execute_tool_call_method_exists(self):
        """Test that execute_tool_call method exists in base class."""
        mock_agent = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.model = Mock()
        mock_agent.model.steps = 1

        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Tool executed"
        mock_message.tool_calls = []
        mock_response.choices = [Mock(message=mock_message)]
        mock_agent.llm.generate.return_value = mock_response

        mock_agent.tool_manager.get_all_tools_schema.return_value = []

        # Create a concrete implementation for testing
        class TestReasoning(Reasoning):
            def plan(self, prompt, obs=None, ttl=1, selected_tools=None):
                return self.execute_tool_call("test message")

        reasoning = TestReasoning(mock_agent)
        plan = reasoning.plan("test")

        assert isinstance(plan, Plan)


class TestIntegration:
    """Integration tests for reasoning components."""

    @pytest.fixture
    def full_mock_agent(self):
        """Create a comprehensive mock agent for integration testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()
        agent.model = Mock()
        agent.model.steps = 10
        agent.unique_id = "integration_test_agent"
        agent.recorder = None
        agent.generate_obs = Mock()

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Integration long term memory"
        agent.memory.format_short_term.return_value = "Integration short term memory"
        agent.memory.add_to_memory = Mock()
        agent.memory.short_term_memory = []

        # Mock _step_display_data as a dictionary to avoid assignment errors
        agent._step_display_data = {}

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = [
            {"function": {"name": "move", "description": "Move to position"}},
            {"function": {"name": "attack", "description": "Attack target"}},
        ]

        # Create mock observation for ReWOO
        mock_obs = Observation(
            step=10,
            self_state={"location": (10, 10), "health": 75, "state": "alert"},
            local_state={
                "enemy_1": {"position": (11, 10), "state": "hostile"},
                "ally_1": {"position": (9, 10), "state": "friendly"},
            },
        )
        agent.generate_obs.return_value = mock_obs

        # Setup different LLM responses for different reasoning types
        def generate_side_effect(*args, **kwargs):
            mock_response = Mock()
            mock_message = Mock()

            # Check if this is a tool execution call (has tool_choice="required")
            if kwargs.get("tool_choice") == "required":
                mock_message.content = "Executing tools"
                mock_message.tool_calls = []
            elif kwargs.get("response_format"):  # ReAct with structured output
                mock_message.content = json.dumps(
                    {
                        "reasoning": "I need to analyze the tactical situation",
                        "action": "Move to safer position",
                    }
                )
            else:
                mock_message.content = "Thinking step by step about the situation"

            mock_response.choices = [Mock(message=mock_message)]
            return mock_response

        agent.llm.generate.side_effect = generate_side_effect
        agent.llm.set_system_prompt = Mock()

        return agent

    def test_all_reasoning_types_with_same_observation(self, full_mock_agent):
        """Test all reasoning types with the same observation and agent."""
        observation = Observation(
            step=10,
            self_state={"location": (10, 10), "health": 75, "state": "alert"},
            local_state={
                "enemy_1": {"position": (11, 10), "state": "hostile"},
                "ally_1": {"position": (9, 10), "state": "friendly"},
            },
        )

        prompt = "Analyze the tactical situation"

        # Test ReAct
        react_reasoning = ReActReasoning(full_mock_agent)
        react_plan = react_reasoning.plan(prompt, observation)
        assert isinstance(react_plan, Plan)
        assert (
            react_plan.step == full_mock_agent.model.steps
        )  # ReAct execute_tool_call uses agent.model.steps

        # Reset mock for next test
        full_mock_agent.llm.generate.side_effect = lambda *args, **kwargs: Mock(
            choices=[Mock(message=Mock(content="CoT thinking process", tool_calls=[]))]
        )

        # Test CoT
        cot_reasoning = CoTReasoning(full_mock_agent)
        cot_plan = cot_reasoning.plan(prompt, observation)
        assert isinstance(cot_plan, Plan)
        assert cot_plan.step == observation.step + 1  # CoT uses obs.step + 1

        # Test ReWOO (doesn't use observation parameter directly)
        rewoo_reasoning = ReWOOReasoning(full_mock_agent)
        rewoo_plan = rewoo_reasoning.plan(prompt)
        assert isinstance(rewoo_plan, Plan)

        # Verify all plans are different instances
        assert react_plan is not cot_plan
        assert cot_plan is not rewoo_plan
        assert react_plan is not rewoo_plan

    def test_selected_tools_across_reasoning_types(self, full_mock_agent):
        """Test selected_tools parameter across different reasoning types."""
        observation = Observation(
            step=5,
            self_state={"location": (5, 5), "health": 100},
            local_state={},
        )

        selected_tools = ["move", "communicate"]
        prompt = "Use specific tools for this task"

        # Test ReAct with selected_tools
        react_reasoning = ReActReasoning(full_mock_agent)
        react_plan = react_reasoning.plan(
            prompt, observation, selected_tools=selected_tools
        )
        assert isinstance(react_plan, Plan)

        # Test CoT with selected_tools
        cot_reasoning = CoTReasoning(full_mock_agent)
        cot_plan = cot_reasoning.plan(
            prompt, observation, selected_tools=selected_tools
        )
        assert isinstance(cot_plan, Plan)

        # Test ReWOO with selected_tools
        rewoo_reasoning = ReWOOReasoning(full_mock_agent)
        rewoo_plan = rewoo_reasoning.plan(prompt, selected_tools=selected_tools)
        assert isinstance(rewoo_plan, Plan)
