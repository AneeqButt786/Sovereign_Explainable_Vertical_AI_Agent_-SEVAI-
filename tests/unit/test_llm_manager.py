"""
Tests for LLM Manager
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.llm_manager import LLMManager

class TestLLMManager:
    @pytest.fixture
    def mock_openai(self):
        with patch('core.llm_manager.OpenAI') as mock_ai:
            yield mock_ai

    def test_generate_success(self, mock_openai):
        """Test successful generation"""
        # Setup mock response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client
        
        llm = LLMManager(api_key="test")
        response = llm.generate("Test prompt")
        
        assert response["output"] == "Test response"
        assert response["total_tokens"] == 15
        assert response["cost"] > 0

    def test_token_counting(self, mock_openai):
        """Test token counting"""
        llm = LLMManager(api_key="test")
        
        # Short text
        assert llm.count_tokens("Hello world") > 0
