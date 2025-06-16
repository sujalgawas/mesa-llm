import os
from collections import deque
from unittest.mock import Mock, patch

import pytest

from mesa_llm.memory import Memory, MemoryEntry


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
    return agent


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    with patch("mesa_llm.memory.ModuleLLM") as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


class TestMemoryEntry:
    """Test the MemoryEntry dataclass"""

    def test_memory_entry_creation(self):
        """Test MemoryEntry creation and basic functionality"""
        entry = MemoryEntry("observation", "Test content", 1, {"key": "value"})

        assert entry.type == "observation"
        assert entry.content == "Test content"
        assert entry.step == 1
        assert entry.metadata == {"key": "value"}


class TestMemory:
    """Test the Memory class core functionality"""

    def test_memory_initialization(self, mock_agent, mock_llm):
        """Test Memory class initialization with defaults and custom values"""
        memory = Memory(
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
        mock_llm.set_system_prompt.assert_called_once()

    @patch("mesa_llm.memory.style")
    @patch("builtins.print")
    def test_add_to_memory(self, mock_print, mock_style, mock_agent, mock_llm):
        """Test adding memories and metadata handling"""
        memory = Memory(agent=mock_agent, api_key="test_key")

        # Test basic addition
        memory.add_to_memory("observation", "Test content")
        assert len(memory.short_term_memory) == 1

        entry = memory.short_term_memory[0]
        assert entry.type == "observation"
        assert entry.content == "Test content"
        assert entry.metadata == {}

        # Test with metadata (note: new interface doesn't support metadata parameter)
        memory.add_to_memory("planning", {"content": "Test plan", "importance": "high"})
        assert len(memory.short_term_memory) == 2
        assert memory.short_term_memory[1].metadata == {"importance": "high"}

        # Test content handling
        memory.add_to_memory("action", {"content": "Test action"})
        assert memory.short_term_memory[2].metadata == {}

    @patch("mesa_llm.memory.style")
    @patch("builtins.print")
    def test_memory_consolidation(self, mock_print, mock_style, mock_agent, mock_llm):
        """Test memory consolidation when capacity is exceeded"""
        mock_llm.generate.return_value = "Consolidated memory summary"

        memory = Memory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=1,
            api_key="test_key",
        )

        # Add memories to trigger consolidation (capacity + consolidation + 1 = 4)
        for i in range(4):
            memory.add_to_memory(f"type_{i}", f"content_{i}")

        # Should have consolidated 1 memory, leaving 3
        assert len(memory.short_term_memory) == 3
        assert memory.long_term_memory == "Consolidated memory summary"
        mock_llm.generate.assert_called_once()

        # Verify the oldest memory was removed
        remaining_contents = [entry.content for entry in memory.short_term_memory]
        assert "content_0" not in remaining_contents

    def test_format_memories(self, mock_agent, mock_llm):
        """Test formatting of short-term and long-term memory"""
        memory = Memory(agent=mock_agent, api_key="test_key")

        # Test empty short-term memory
        assert memory.format_short_term() == "No recent memory."

        # Test with entries
        memory.short_term_memory.append(MemoryEntry("observation", "Test obs", 1, {}))
        memory.short_term_memory.append(MemoryEntry("planning", "Test plan", 2, {}))

        result = memory.format_short_term()
        assert "[TestAgent(123) Short-Term Memory]" in result
        assert "[Observation @ Step 1]" in result
        assert "Test obs" in result
        assert "[Planning @ Step 2]" in result
        assert "Test plan" in result

        # Test long-term memory formatting
        memory.long_term_memory = "Long-term summary"
        assert memory.format_long_term() == "Long-term summary"

    def test_convert_entry_to_dict(self, mock_agent, mock_llm):
        """Test memory entry conversion to dictionary"""
        memory = Memory(agent=mock_agent, api_key="test_key")
        entry = MemoryEntry("observation", "Test content", 1, {"key": "value"})

        result = memory.convert_entry_to_dict(entry)
        expected = {
            "type": "observation",
            "content": "Test content",
            "step": 1,
            "metadata": {"key": "value"},
        }
        assert result == expected

    def test_update_long_term_memory(self, mock_agent, mock_llm):
        """Test long-term memory update process"""
        mock_llm.generate.return_value = "Updated long-term memory"

        memory = Memory(agent=mock_agent, api_key="test_key")
        memory.long_term_memory = "Previous memory"

        memories = [
            MemoryEntry("obs", "Test observation", 1, {}),
            MemoryEntry("plan", "Test planning", 2, {}),
        ]

        memory._update_long_term_memory(memories)

        # Verify LLM was called with correct prompt structure
        call_args = mock_llm.generate.call_args[0][0]
        assert "Short term memory:" in call_args
        assert "Long term memory:" in call_args
        assert "Previous memory" in call_args

        assert memory.long_term_memory == "Updated long-term memory"
