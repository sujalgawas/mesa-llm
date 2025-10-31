import json
from collections import deque
from unittest.mock import MagicMock

import pytest

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.memory import MemoryEntry


@pytest.fixture
def mock_agent():
    agent = MagicMock(name="MockLLMAgent")

    # Create a MagicMock for the LLM's response
    mock_response = MagicMock()

    # This line *defines* the full nested path on the mock
    mock_response.choices[0].message.content = json.dumps({"grade": 3})

    # Set this as the return value
    agent.llm.generate.return_value = mock_response

    agent.model.steps = 100
    return agent


class TestEpisodicMemory:
    """Core functionality test"""

    def test_memory_init(self, mock_agent):
        """Test EpisodicMemory class initialization with defaults and custom values"""
        memory = EpisodicMemory(
            agent=mock_agent,
            max_capacity=10,
            considered_entries=5,
            llm_model="provider/test_model",
        )

        assert memory.agent == mock_agent
        assert memory.max_capacity == 10
        assert memory.considered_entries == 5
        assert isinstance(memory.memory_entries, deque)
        assert memory.memory_entries.maxlen == 10
        assert memory.system_prompt is not None
        """FYI: The above line may not always work; use the one below if needed."""
        # assert isinstance(memory.system_prompt,str), memory.system_prompt.strip() != ""

    def test_add_memory_entry(self, mock_agent):
        """Test adding memories to Episodic memory"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # Should be empty step_content initially
        assert memory.step_content != {}

    def test_grade_event_importance(self, mock_agent):
        """Test grading event importance"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        # 1. Set up a specific grade for this test
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"grade": 5})
        mock_agent.llm.generate.return_value = mock_response

        # 2. Call the method
        grade = memory.grade_event_importance("observation", {"data": "critical info"})

        # 3. Assert the grade is correct
        assert grade == 5

        # 4. Assert the LLM was called correctly
        mock_agent.llm.generate.assert_called_once()

        # Check that the system prompt was set on the llm object
        assert memory.llm.system_prompt == memory.system_prompt

    def test_retrieve_top_k_entries(self, mock_agent):
        """Test the sorting logic for retrieving entries (importance - recency_penalty)."""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        # Set current step
        mock_agent.model.steps = 100

        # Manually add entries to bypass grading and control scores
        # score = importance - (current_step - entry_step)

        # score = 5 - (100 - 98) = 3
        entry_a = MemoryEntry(
            content={"importance": 5, "id": "A"}, step=98, agent=mock_agent
        )
        # score = 1 - (100 - 99) = 0
        entry_b = MemoryEntry(
            content={"importance": 1, "id": "B"}, step=99, agent=mock_agent
        )
        # score = 4 - (100 - 90) = -6
        entry_c = MemoryEntry(
            content={"importance": 4, "id": "C"}, step=90, agent=mock_agent
        )
        # score = 4 - (100 - 95) = -1
        entry_d = MemoryEntry(
            content={"importance": 4, "id": "D"}, step=95, agent=mock_agent
        )

        memory.memory_entries.extend([entry_a, entry_b, entry_c, entry_d])

        # Retrieve top 3 (k=3)
        top_entries = memory.retrieve_top_k_entries(3)

        # Expected order: A (3), B (0), D (-1)
        assert len(top_entries) == 3
        assert top_entries[0].content["id"] == "A"
        assert top_entries[1].content["id"] == "B"
        assert top_entries[2].content["id"] == "D"

        # Entry C (score -6) should be omitted
        assert "C" not in [e.content["id"] for e in top_entries]
