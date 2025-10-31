from unittest.mock import patch

from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import MemoryEntry


class TestLTMemory:
    """Test the Memory class core functionality"""

    def test_memory_initialization(self, mock_agent):
        """Test Memory class initialization with defaults and custom values"""
        memory = LongTermMemory(
            agent=mock_agent,
            llm_model="provider/test_model",
        )

        assert memory.agent == mock_agent
        assert memory.long_term_memory == ""
        assert memory.llm.system_prompt is not None

    def test_update_long_term_memory(self, mock_agent, mock_llm):
        """Test updating long-term memory functionality"""
        # Mock the LLM's generate method
        mock_llm.generate.return_value = "Updated long-term memory"

        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        # Replace the real LLM with our mock
        memory.llm = mock_llm
        memory.long_term_memory = "Previous memory"

        # Add some content to buffer
        memory.buffer = MemoryEntry(
            agent=mock_agent,
            content={"message": "Test message"},
            step=1,
        )

        memory._update_long_term_memory()

        # Verify LLM can call with correct prompt structure
        call_args = mock_llm.generate.call_args[0][0]
        assert "new memory entry" in call_args
        assert "Long term memory" in call_args

        assert memory.long_term_memory == "Updated long-term memory"

    # process step test
    def test_process_step(self, mock_agent):
        """Test process_step functionality"""
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")

        # Add some content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Process the step
        with (
            patch("rich.console.Console"),
            patch.object(memory.llm, "generate", return_value="mocked summary"),
        ):
            memory.process_step(pre_step=True)
            assert isinstance(memory.buffer, MemoryEntry)
            # assert memory.buffer is not None

            # Process post-step
            memory.process_step(pre_step=False)
            assert memory.long_term_memory == "mocked summary"

    # format memories test
    def test_format_long_term(self, mock_agent):
        """Test formatting long-term memory"""
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.long_term_memory = "Long-term summary"

        assert memory.format_long_term() == "Long-term summary"
