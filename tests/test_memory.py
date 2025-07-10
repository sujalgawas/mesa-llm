import os
from collections import deque
from unittest.mock import Mock, patch

import pytest

from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.memory.st_lt_memory import STLTMemory


@pytest.fixture(autouse=True)
def mock_environment():
    """Ensure tests don't depend on real environment variables"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        yield


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    agent = Mock()
    agent.__class__.__name__ = "TestAgent"
    agent.unique_id = 123
    agent.__str__ = Mock(return_value="TestAgent(123)")
    agent.model = Mock()
    agent.model.steps = 1
    return agent


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    with patch("mesa_llm.module_llm.ModuleLLM") as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


class TestMemoryEntry:
    """Test the MemoryEntry dataclass"""

    def test_memory_entry_creation(self):
        """Test MemoryEntry creation and basic functionality"""
        content = {"observation": "Test content", "metadata": "value"}
        entry = MemoryEntry(content=content, step=1)

        assert entry.content == content
        assert entry.step == 1

    def test_memory_entry_str(self):
        """Test MemoryEntry string representation"""
        content = {"observation": "Test content", "type": "observation"}
        entry = MemoryEntry(content=content, step=1)

        str_repr = str(entry)
        assert "Test content" in str_repr
        assert "observation" in str_repr


class TestMemory:
    """Test the Memory class core functionality"""

    def test_memory_initialization(self, mock_agent, mock_llm):
        """Test Memory class initialization with defaults and custom values"""
        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=3,
            consolidation_capacity=1,
            api_key="test_key",
            llm_model="test_model",
        )

        assert memory.agent == mock_agent
        assert memory.capacity == 3
        assert memory.consolidation_capacity == 1
        assert isinstance(memory.short_term_memory, deque)
        assert memory.long_term_memory == ""
        assert memory.llm.system_prompt is not None

    def test_add_to_memory(self, mock_agent, mock_llm):
        """Test adding memories to short-term memory"""
        memory = STLTMemory(agent=mock_agent, api_key="test_key")

        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # Should be empty step_content initially
        assert memory.step_content != {}

    def test_process_step(self, mock_agent, mock_llm):
        """Test process_step functionality"""
        memory = STLTMemory(agent=mock_agent, api_key="test_key")

        # Add some content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Process the step
        with patch("rich.console.Console"):
            memory.process_step(pre_step=True)
            assert len(memory.short_term_memory) == 1

            # Process post-step
            memory.process_step(pre_step=False)

    def test_memory_consolidation(self, mock_agent, mock_llm):
        """Test memory consolidation when capacity is exceeded"""
        mock_llm.generate.return_value = "Consolidated memory summary"

        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=1,
            api_key="test_key",
        )

        # Add memories to trigger consolidation
        with patch("rich.console.Console"):
            for i in range(5):
                memory.add_to_memory("observation", {"content": f"content_{i}"})
                memory.process_step(pre_step=True)
                memory.process_step(pre_step=False)

        # Should have consolidated some memories
        assert (
            len(memory.short_term_memory)
            <= memory.capacity + memory.consolidation_capacity
        )

    def test_format_memories(self, mock_agent, mock_llm):
        """Test formatting of short-term and long-term memory"""
        memory = STLTMemory(agent=mock_agent, api_key="test_key")

        # Test empty short-term memory
        assert memory.format_short_term() == "No recent memory."

        # Test with entries
        memory.short_term_memory.append(
            MemoryEntry(content={"observation": "Test obs"}, step=1)
        )
        memory.short_term_memory.append(
            MemoryEntry(content={"planning": "Test plan"}, step=2)
        )

        result = memory.format_short_term()
        assert "[TestAgent(123) Short-Term Memory]" in result
        assert "[Step 1]" in result
        assert "Test obs" in result
        assert "[Step 2]" in result
        assert "Test plan" in result

        # Test long-term memory formatting
        memory.long_term_memory = "Long-term summary"
        assert memory.format_long_term() == "Long-term summary"

    def test_update_long_term_memory(self, mock_agent, mock_llm):
        """Test long-term memory update process"""
        mock_llm.generate.return_value = "Updated long-term memory"

        memory = STLTMemory(agent=mock_agent, api_key="test_key")
        # Replace the real LLM with our mock
        memory.llm = mock_llm
        memory.long_term_memory = "Previous memory"

        memory._update_long_term_memory()

        # Verify LLM was called with correct prompt structure
        call_args = mock_llm.generate.call_args[0][0]
        assert "Short term memory:" in call_args
        assert "Long term memory:" in call_args
        assert "Previous memory" in call_args

        assert memory.long_term_memory == "Updated long-term memory"

    def test_observation_tracking(self, mock_agent, mock_llm):
        """Test that observations are properly tracked and only changes stored"""
        memory = STLTMemory(agent=mock_agent, api_key="test_key")

        # First observation
        obs1 = {"position": (0, 0), "health": 100}
        memory.add_to_memory("observation", obs1)

        # Same observation (should not add much to step_content)
        memory.add_to_memory("observation", obs1)

        # Changed observation
        obs2 = {"position": (1, 1), "health": 90}
        memory.add_to_memory("observation", obs2)

        # Verify last observation is tracked
        assert memory.last_observation == obs2
