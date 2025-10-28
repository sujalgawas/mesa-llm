from collections import deque
from unittest.mock import patch 

from mesa_llm.memory.memory import MemoryEntry  
from mesa_llm.memory.episodic_memory import EpisodicMemory


class TestEpisodicMemory:
    """Core functionality test"""
    
    def test_memory_init(self, mock_agent):
        memory = EpisodicMemory(
            agent=mock_agent,
            max_capacity = 10,
            considered_entries = 5,
            llm_model="provider/test_model"
        )
        
        assert memory.agent == mock_agent
        assert memory.max_capacity == 10
        assert memory.considered_entries == 5
        assert isinstance(memory.memory_entries, deque)
        assert memory.memory_entries.maxlen == 10
        assert memory.system_prompt is not None
        """FYI: The above line may not always work; use the one below if needed."""
        #assert isinstance(memory.system_prompt,str), memory.system_prompt.strip() != ""
        
            
