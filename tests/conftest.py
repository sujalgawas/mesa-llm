import os
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_environment():
    """Ensure tests don't depend on real environment variables"""
    with patch.dict(
        os.environ,
        {"PROVIDER_API_KEY": "test_key", "OPENAI_API_KEY": "test_openai_key"},
        clear=True,
    ):
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
    agent.step_prompt = "Test step prompt"
    return agent


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    with patch("mesa_llm.module_llm.ModuleLLM") as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance
