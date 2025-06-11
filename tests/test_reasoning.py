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
        assert "[Self State]" in str_repr
        assert "[Local State of Nearby Agents]" in str_repr
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


class TestReActReasoning:
    """Test cases for ReActReasoning class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Long term memory content"
        agent.memory.format_short_term.return_value = "Short term memory content"
        agent.memory.add_to_memory = Mock()

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = {"tools": "schema"}

        # Setup LLM mock
        mock_response = Mock()
        mock_message = Mock()
        mock_message = "I will move to position (2, 3)"
        mock_response.choices = [Mock(message=mock_message)]
        agent.llm.generate.return_value = mock_response
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
        assert plan.step == sample_observation.step + 1
        assert plan.ttl == 1

        # Verify LLM interactions
        mock_agent.llm.set_system_prompt.assert_called_once()
        mock_agent.llm.generate.assert_called_once_with(
            prompt=prompt, tool_schema={"tools": "schema"}
        )

        # Verify memory interactions
        assert mock_agent.memory.add_to_memory.call_count == 2  # observation + plan

        # Check system prompt content
        system_prompt_call = mock_agent.llm.set_system_prompt.call_args[0][0]
        assert "Long-Term Memory" in system_prompt_call
        assert "Short-Term Memory" in system_prompt_call
        assert "Current Observation" in system_prompt_call


class TestCoTReasoning:
    """Test cases for CoTReasoning class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Long term memory content"
        agent.memory.format_short_term.return_value = "Short term memory content"
        agent.memory.add_to_memory = Mock()

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = {"tools": "schema"}

        # Setup LLM mock
        mock_response = Mock()
        mock_message = Mock()
        mock_message = "Thought 1: I see an enemy\nAction: Move away"
        mock_response.choices = [Mock(message=mock_message)]
        agent.llm.generate.return_value = mock_response
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
        assert plan.step == sample_observation.step + 1
        assert plan.ttl == 1

        # Verify system prompt contains CoT instructions
        system_prompt_call = mock_agent.llm.set_system_prompt.call_args[0][0]
        assert "Chain-of-Thought" in system_prompt_call
        assert "Thought 1:" in system_prompt_call

        # Verify memory interactions
        assert mock_agent.memory.add_to_memory.call_count == 2


class TestReWOOReasoning:
    """Test cases for ReWOOReasoning class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.llm = Mock()
        agent.memory = Mock()
        agent.tool_manager = Mock()

        # Setup memory mock methods
        agent.memory.format_long_term.return_value = "Long term memory content"
        agent.memory.format_short_term.return_value = "Short term memory content"
        agent.memory.add_to_memory = Mock()

        # Setup tool manager mock
        agent.tool_manager.get_all_tools_schema.return_value = {"tools": "schema"}

        # Setup LLM mock
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Plan: Multi-step strategy\nStep 1: Scout area"
        mock_response.choices = [Mock(message=mock_message)]
        agent.llm.generate.return_value = mock_response
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

    def test_rewoo_reasoning_plan(self, mock_agent, sample_observation):
        """Test ReWOOReasoning plan method."""
        reasoning = ReWOOReasoning(mock_agent)
        prompt = "Create a comprehensive multi-step plan"
        ttl = 3

        plan = reasoning.plan(prompt, sample_observation, ttl=ttl)

        # Verify the plan was created correctly
        assert isinstance(plan, Plan)
        assert plan.step == sample_observation.step + 1
        assert plan.ttl == ttl

        # Verify system prompt contains ReWOO instructions
        system_prompt_call = mock_agent.llm.set_system_prompt.call_args[0][0]
        assert "ReWOO" in system_prompt_call
        assert "multi-step plan" in system_prompt_call
        assert "Step 1:" in system_prompt_call
        assert "Step 2:" in system_prompt_call
        assert "Contingency:" in system_prompt_call

        # Verify memory interactions
        assert mock_agent.memory.add_to_memory.call_count == 2

    def test_rewoo_reasoning_plan_default_ttl(self, mock_agent, sample_observation):
        """Test ReWOOReasoning plan method with default TTL."""
        reasoning = ReWOOReasoning(mock_agent)
        prompt = "Create a plan"

        plan = reasoning.plan(prompt, sample_observation)

        # TTL should default to the parameter default, not the Plan default
        assert plan.ttl == 1


class TestReasoningAbstractClass:
    """Test cases for the abstract Reasoning class."""

    def test_reasoning_is_abstract(self):
        """Test that Reasoning cannot be instantiated directly."""
        mock_agent = Mock()

        with pytest.raises(TypeError):
            Reasoning(mock_agent)

    def test_reasoning_subclass_must_implement_plan(self):
        """Test that subclasses must implement the plan method."""

        class IncompleteReasoning(Reasoning):
            def __init__(self, agent):
                super().__init__(agent)

        mock_agent = Mock()

        with pytest.raises(TypeError):
            IncompleteReasoning(mock_agent)


class TestIntegration:
    """Integration tests for the reasoning system."""

    @pytest.fixture
    def full_mock_agent(self):
        """Create a comprehensive mock agent."""
        agent = Mock()

        # LLM mock
        agent.llm = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message = "Integration test response"
        mock_response.choices = [Mock(message=mock_message)]
        agent.llm.generate.return_value = mock_response

        # Memory mock
        agent.memory = Mock()
        agent.memory.format_long_term.return_value = "Integration long term memory"
        agent.memory.format_short_term.return_value = "Integration short term memory"

        # Tool manager mock
        agent.tool_manager = Mock()
        agent.tool_manager.get_all_tools_schema.return_value = {
            "teleport_to_location": {"type": "function"},
            "speak_to": {"type": "function"},
        }

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
        assert react_plan.step == 11

        # Test CoT
        cot_reasoning = CoTReasoning(full_mock_agent)
        cot_plan = cot_reasoning.plan(prompt, observation)
        assert isinstance(cot_plan, Plan)
        assert cot_plan.step == 11

        # Test ReWOO
        rewoo_reasoning = ReWOOReasoning(full_mock_agent)
        rewoo_plan = rewoo_reasoning.plan(prompt, observation, ttl=5)
        assert isinstance(rewoo_plan, Plan)
        assert rewoo_plan.step == 11
        assert rewoo_plan.ttl == 5

        # Verify all reasoning types called the agent's methods
        assert full_mock_agent.llm.generate.call_count == 3
        assert (
            full_mock_agent.memory.add_to_memory.call_count == 6
        )  # 2 calls per reasoning type
