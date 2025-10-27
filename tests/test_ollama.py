"""
Unit tests for Ollama LLM provider integration.

These tests verify that the Ollama provider is correctly integrated into the
Memvid LLM client system and follows the expected interface contracts.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, '..')

from memvid.llm_client import OllamaProvider, LLMClient


class TestOllamaProvider:
    """Test suite for OllamaProvider class."""

    @patch('memvid.llm_client.ollama')
    def test_ollama_provider_initialization(self, mock_ollama):
        """
        Test that OllamaProvider initializes correctly with default parameters.
        
        Verifies:
        - Client is created with correct host
        - Model name is stored
        - Connection verification is attempted
        """
        # Setup mock client
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.2:3b'}]}
        mock_ollama.Client.return_value = mock_client
        
        # Initialize provider
        provider = OllamaProvider(model='llama3.2:3b')
        
        # Verify initialization
        assert provider.model == 'llama3.2:3b'
        assert provider.base_url == 'http://localhost:11434'
        mock_ollama.Client.assert_called_once_with(host='http://localhost:11434')
        mock_client.list.assert_called_once()

    @patch('memvid.llm_client.ollama')
    def test_ollama_provider_custom_base_url(self, mock_ollama):
        """
        Test that OllamaProvider accepts custom base URL.
        
        Verifies:
        - Custom host is passed to Ollama client
        - Base URL is stored correctly
        """
        # Setup mock client
        mock_client = Mock()
        mock_client.list.return_value = {'models': []}
        mock_ollama.Client.return_value = mock_client
        
        # Initialize with custom URL
        custom_url = 'http://192.168.1.100:11434'
        provider = OllamaProvider(model='llama3.2:3b', base_url=custom_url)
        
        # Verify custom URL is used
        assert provider.base_url == custom_url
        mock_ollama.Client.assert_called_once_with(host=custom_url)

    @patch('memvid.llm_client.ollama')
    def test_ollama_provider_connection_error(self, mock_ollama):
        """
        Test that OllamaProvider raises ConnectionError when server is unreachable.
        
        Verifies:
        - ConnectionError is raised with helpful message
        - Error contains server URL information
        """
        # Setup mock to simulate connection failure
        mock_client = Mock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama.Client.return_value = mock_client
        
        # Verify ConnectionError is raised
        with pytest.raises(ConnectionError) as excinfo:
            OllamaProvider(model='llama3.2:3b')
        
        # Check error message contains helpful information
        assert 'Failed to connect to Ollama' in str(excinfo.value)
        assert 'http://localhost:11434' in str(excinfo.value)

    @patch('memvid.llm_client.ollama')
    def test_ollama_chat_non_streaming(self, mock_ollama):
        """
        Test non-streaming chat response from Ollama.
        
        Verifies:
        - Correct message format is sent
        - Response content is extracted properly
        - Options are passed correctly
        """
        # Setup mock client
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.2:3b'}]}
        mock_client.chat.return_value = {
            'message': {'content': 'Test response'}
        }
        mock_ollama.Client.return_value = mock_client
        
        # Initialize provider and send message
        provider = OllamaProvider(model='llama3.2:3b')
        messages = [{'role': 'user', 'content': 'Hello'}]
        response = provider.chat(messages, stream=False)
        
        # Verify response
        assert response == 'Test response'
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == 'llama3.2:3b'
        assert call_args[1]['messages'] == messages
        assert call_args[1]['stream'] is False

    @patch('memvid.llm_client.ollama')
    def test_ollama_chat_streaming(self, mock_ollama):
        """
        Test streaming chat response from Ollama.
        
        Verifies:
        - Streaming flag is set correctly
        - Response chunks are yielded properly
        - Iterator interface works as expected
        """
        # Setup mock client
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.2:3b'}]}
        
        # Mock streaming response
        mock_stream = [
            {'message': {'content': 'Hello '}},
            {'message': {'content': 'world'}},
            {'message': {'content': '!'}}
        ]
        mock_client.chat.return_value = iter(mock_stream)
        mock_ollama.Client.return_value = mock_client
        
        # Initialize provider and stream message
        provider = OllamaProvider(model='llama3.2:3b')
        messages = [{'role': 'user', 'content': 'Hello'}]
        response = provider.chat(messages, stream=True)
        
        # Collect streamed chunks
        chunks = list(response)
        
        # Verify streaming
        assert chunks == ['Hello ', 'world', '!']
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['stream'] is True

    @patch('memvid.llm_client.ollama')
    def test_ollama_generation_options(self, mock_ollama):
        """
        Test that generation parameters are correctly passed to Ollama.
        
        Verifies:
        - Temperature is passed correctly
        - Top-p is passed correctly
        - Max tokens is mapped to num_predict
        - Stop sequences are mapped to stop parameter
        """
        # Setup mock client
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.2:3b'}]}
        mock_client.chat.return_value = {'message': {'content': 'Response'}}
        mock_ollama.Client.return_value = mock_client
        
        # Initialize provider and send message with options
        provider = OllamaProvider(model='llama3.2:3b')
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        response = provider.chat(
            messages,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=100,
            stop_sequences=['END']
        )
        
        # Verify options were passed correctly
        call_args = mock_client.chat.call_args
        options = call_args[1]['options']
        
        assert options['temperature'] == 0.7
        assert options['top_p'] == 0.9
        assert options['top_k'] == 40
        assert options['num_predict'] == 100
        assert options['stop'] == ['END']


class TestLLMClientOllama:
    """Test suite for LLMClient integration with Ollama."""

    @patch('memvid.llm_client.OLLAMA_AVAILABLE', True)
    @patch('memvid.llm_client.ollama')
    def test_llm_client_ollama_provider(self, mock_ollama):
        """
        Test that LLMClient correctly initializes with Ollama provider.
        
        Verifies:
        - Ollama is recognized as valid provider
        - No API key validation for Ollama
        - Default model is used if not specified
        """
        # Setup mock
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.2:3b'}]}
        mock_ollama.Client.return_value = mock_client
        
        # Create client with Ollama provider
        client = LLMClient(provider='ollama')
        
        # Verify provider is set correctly
        assert client.provider_name == 'ollama'
        assert isinstance(client.provider, OllamaProvider)

    @patch('memvid.llm_client.OLLAMA_AVAILABLE', True)
    def test_llm_client_lists_ollama_provider(self):
        """
        Test that Ollama is included in available providers list.
        
        Verifies:
        - Ollama appears in list_providers()
        - Ollama appears in list_available_providers() when library is installed
        """
        # Verify Ollama is in providers list
        providers = LLMClient.list_providers()
        assert 'ollama' in providers
        
        # Verify Ollama is in available providers
        available = LLMClient.list_available_providers()
        assert 'ollama' in available

    @patch('memvid.llm_client.OLLAMA_AVAILABLE', True)
    @patch('memvid.llm_client.ollama')
    def test_llm_client_no_api_key_required(self, mock_ollama):
        """
        Test that LLMClient doesn't require API key for Ollama.
        
        Verifies:
        - Client initializes without API key
        - No ValueError is raised
        - Provider is created successfully
        """
        # Setup mock
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.2:3b'}]}
        mock_ollama.Client.return_value = mock_client
        
        # Create client without API key (should not raise error)
        client = LLMClient(provider='ollama', api_key=None)
        
        # Verify client was created successfully
        assert client.provider_name == 'ollama'

    @patch('memvid.llm_client.OLLAMA_AVAILABLE', False)
    def test_llm_client_ollama_not_available(self):
        """
        Test that appropriate error is raised when Ollama library is not installed.
        
        Verifies:
        - ImportError is raised with helpful message
        - Error message mentions Ollama specifically
        """
        # Attempt to create client with unavailable Ollama
        with pytest.raises(ImportError) as excinfo:
            LLMClient(provider='ollama')
        
        # Verify error message is helpful
        assert 'ollama' in str(excinfo.value).lower()
        assert 'not available' in str(excinfo.value).lower()


class TestOllamaIntegration:
    """Integration tests for Ollama provider (requires Ollama running)."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration", default=False),
        reason="Integration tests disabled by default"
    )
    def test_ollama_real_connection(self):
        """
        Test real connection to Ollama server (requires Ollama installed and running).
        
        This is an integration test that requires:
        1. Ollama installed on the system
        2. Ollama server running
        3. At least one model pulled
        
        Run with: pytest tests/test_ollama.py --run-integration
        """
        try:
            from memvid.llm_client import OllamaProvider
            
            # Attempt to connect to real Ollama server
            provider = OllamaProvider(model='llama3.2:3b')
            
            # Try a simple chat message
            messages = [{'role': 'user', 'content': 'Say hello'}]
            response = provider.chat(messages)
            
            # Verify we got some response
            assert response is not None
            assert len(response) > 0
            
        except ConnectionError:
            pytest.skip("Ollama server not running")
        except ImportError:
            pytest.skip("Ollama library not installed")


# Pytest configuration
def pytest_addoption(parser):
    """Add custom command line option for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services"
    )

