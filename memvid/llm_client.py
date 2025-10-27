
# memvid/llm_client.py

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator

from memvid.config import DEFAULT_LLM_MODELS

# Optional imports with availability checking
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. OpenAI provider will be disabled.")

try:
    import google as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: Google Generative AI library not available. Google provider will be disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic library not available. Anthropic provider will be disabled.")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: Ollama library not available. Ollama provider will be disabled.")

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """Send chat messages and get response"""
        pass

    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Stream chat response"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """Send chat messages to OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs
            )

            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Stream chat response from OpenAI"""
        return self.chat(messages, stream=True, **kwargs)

    def _stream_response(self, response) -> Iterator[str]:
        """Process streaming response from OpenAI"""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class GoogleProvider(LLMProvider):
    """Google provider implementation based on your working code"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.llm_client = genai.Client(api_key=api_key)
        self.model_name = model
        self.api_key = api_key

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """Send chat messages to Google using your proven patterns"""
        try:
            # Convert to Google message format
            gemini_messages = self._convert_messages_to_gemini(messages)

            # Safety settings (from your working code)
            if 'safety_settings' not in kwargs:
                kwargs['safety_settings'] = {
                    'HARASSMENT': 'block_none',
                    'DANGEROUS': 'block_none',
                    'SEXUAL': 'block_none',
                    'HATE_SPEECH': 'block_none'
                }
            
            # Get generation config
            generation_config = self._extract_generation_config(kwargs)

            if stream:
                response = self.llm_client.models.generate_content_stream(
                    model=self.model_name,
                    contents=gemini_messages,
                    config=generation_config,
                )
                return self._stream_response(response)
            else:
                response = self.llm_client.models.generate_content(
                    model=self.model_name,
                    contents=gemini_messages,
                    config=generation_config,
                )
                return response.text

        except Exception as e:
            print(f"Google API error: {e}")
            return None

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Stream chat response from Google"""
        return self.chat(messages, stream=True, **kwargs)

    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """Convert OpenAI format to Google format using your working patterns"""
        gemini_messages = []

        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')

            # Convert role names to Google format
            if role == 'assistant':
                role = 'model'
            elif role == 'system':
                # Handle system messages by prepending to first user message
                if gemini_messages and gemini_messages[-1]['role'] == 'user':
                    gemini_messages[-1]['parts'][0]['text'] = f"{content}\n\n{gemini_messages[-1]['parts'][0]['text']}"
                else:
                    gemini_messages.append({
                        'role': 'user',
                        'parts': [{'text': content}]
                    })
                continue

            gemini_messages.append({
                'role': role,
                'parts': [{'text': content}]
            })

        return gemini_messages

    def _extract_generation_config(self, kwargs: Dict[str, Any]):
        """Extract Google-compatible generation config from kwargs"""
        from google.genai.types import GenerateContentConfig
        config_params = {}

        # Map common parameters to Google format
        if 'temperature' in kwargs:
            config_params['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            config_params['max_output_tokens'] = kwargs['max_tokens']
        if 'top_p' in kwargs:
            config_params['top_p'] = kwargs['top_p']
        if 'stop_sequences' in kwargs:
            config_params['stop_sequences'] = kwargs['stop_sequences']
        if 'safety_settings' in kwargs:
            config_params['safety_settings'] = kwargs['safety_settings']

        return GenerateContentConfig(**config_params) if config_params else None

    def _stream_response(self, response) -> Iterator[str]:
        """Process streaming response from Google using your working approach"""
        accumulated_text = ""

        for chunk in response:
            chunk_text = ""
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                chunk_text += part.text

            if chunk_text:
                accumulated_text += chunk_text
                yield chunk_text

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation based on your working code"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.api_key = api_key

        # Special handling for reasoning models
        self.is_reasoning_model = "claude-3-7-sonnet" in model


    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """Send chat messages to Anthropic using your proven patterns"""
        try:
            # Convert to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)
            system_prompt = self._extract_system_prompt(messages)


            # Get generation config
            max_tokens = kwargs.get('max_tokens', 8096)
            temperature = kwargs.get('temperature', 0.7)
            top_p = kwargs.get('top_p', 0.9)
            stop_sequences = kwargs.get('stop_sequences', None)

            if stream:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_prompt,
                    messages=anthropic_messages,
                    stop_sequences=stop_sequences,
                    stream=True
                )
                return self._stream_response(response)
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_prompt,
                    messages=anthropic_messages,
                    stop_sequences=stop_sequences
                )
                return response.content[0].text

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Stream chat response from Anthropic"""
        return self.chat(messages, stream=True, **kwargs)

    def _convert_messages_to_anthropic(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """Convert OpenAI format to Anthropic format"""
        anthropic_messages = []

        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')

            # Skip system messages (handled separately)
            if role == 'system':
                continue

            # Convert assistant to Claude's expected format
            if role == 'assistant':
                anthropic_messages.append({
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': content}]
                })
            else:
                anthropic_messages.append({
                    'role': 'user',
                    'content': [{'type': 'text', 'text': content}]
                })

        return anthropic_messages

    def _extract_system_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract system prompt from messages"""
        system_messages = [msg['content'] for msg in messages if msg.get('role') == 'system']
        return '\n\n'.join(system_messages) if system_messages else ""

    def _stream_response(self, response) -> Iterator[str]:
        """Process streaming response from Anthropic using your working approach"""
        accumulated_text = ""

        for chunk in response:
            chunk_text = ""

            if chunk.type == "content_block_delta":
                chunk_text = chunk.delta.text
                accumulated_text += chunk_text

                if chunk_text:
                    yield chunk_text

            elif chunk.type == "message_stop":
                break

class OllamaProvider(LLMProvider):
    """
    Ollama provider implementation for local LLM models.
    
    Ollama allows running open-source language models locally without API keys.
    Models like Llama, Mistral, Phi, and others can be run on your own hardware.
    
    Requirements:
        - Ollama must be installed and running locally (default: http://localhost:11434)
        - Install with: `curl -fsSL https://ollama.com/install.sh | sh`
        - Pull models with: `ollama pull <model-name>`
        
    Example models:
        - llama3.2:3b - Small, fast Llama model
        - llama3.2:latest - Latest Llama model
        - mistral:latest - Mistral 7B model
        - phi3:latest - Microsoft Phi-3 model
        - codellama:latest - Code-specialized Llama
    """

    def __init__(self, api_key: str = None, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider for local model inference.
        
        Args:
            api_key: Not used for Ollama (local models don't need API keys), kept for interface compatibility
            model: Name of the Ollama model to use (e.g., 'llama3.2:3b', 'mistral:latest')
            base_url: URL where Ollama server is running (default: http://localhost:11434)
        """
        # Import ollama client library
        import ollama
        
        # Store the Ollama client instance
        self.client = ollama.Client(host=base_url)
        
        # Store model name for inference requests
        self.model = model
        
        # Store base URL for connection management
        self.base_url = base_url
        
        # Verify connection to Ollama server and check if model is available
        self._verify_connection()

    def _verify_connection(self):
        """
        Verify that Ollama server is running and the specified model is available.
        
        Raises:
            ConnectionError: If Ollama server is not reachable
            ValueError: If specified model is not pulled/available
        """
        try:
            # Attempt to list available models to verify connection
            available_models = self.client.list()
            
            # Extract model names from the response
            # Ollama returns a dict with 'models' key containing a list of model objects
            model_names = []
            if available_models and 'models' in available_models:
                for model in available_models['models']:
                    # Model can be a dict or an object, handle both cases
                    if isinstance(model, dict):
                        # If it's a dict, try both 'name' and 'model' keys
                        model_name = model.get('name') or model.get('model', 'Unknown')
                    else:
                        # If it's an object, try to get the name attribute
                        model_name = getattr(model, 'name', None) or getattr(model, 'model', 'Unknown')
                    model_names.append(model_name)
            
            # Check if our model is in the available models list
            if self.model not in model_names:
                # Provide helpful error message with available models
                print(f"Warning: Model '{self.model}' not found locally.")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                print(f"To pull this model, run: ollama pull {self.model}")
                
        except Exception as e:
            # Connection failed - Ollama server might not be running
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is installed and running. Error: {e}"
            )

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """
        Send chat messages to Ollama local model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Format: [{'role': 'user'|'assistant'|'system', 'content': 'text'}]
            stream: If True, returns an iterator for streaming responses
            **kwargs: Additional parameters for Ollama (temperature, top_p, etc.)
        
        Returns:
            String response if stream=False, Iterator[str] if stream=True
            
        Note:
            Ollama uses the same message format as OpenAI, so no conversion is needed
        """
        try:
            # Extract common generation parameters from kwargs
            options = self._extract_generation_options(kwargs)
            
            # Make request to Ollama with the specified model
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                options=options
            )

            # Handle streaming vs non-streaming response
            if stream:
                return self._stream_response(response)
            else:
                # Extract text content from the response message
                return response['message']['content']

        except Exception as e:
            # Log error and return None to maintain consistent error handling
            print(f"Ollama API error: {e}")
            return None

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Stream chat response from Ollama local model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for Ollama (temperature, top_p, etc.)
            
        Returns:
            Iterator yielding text chunks as they are generated
        """
        # Delegate to chat method with streaming enabled
        return self.chat(messages, stream=True, **kwargs)

    def _extract_generation_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and convert generation parameters to Ollama format.
        
        Ollama supports various generation options that control model behavior:
        - temperature: Controls randomness (0.0 = deterministic, 1.0+ = creative)
        - top_p: Nucleus sampling threshold (0.0-1.0)
        - top_k: Top-k sampling limit
        - num_predict: Maximum tokens to generate
        - stop: Stop sequences to end generation
        
        Args:
            kwargs: Dictionary of potential generation parameters
            
        Returns:
            Dictionary with Ollama-compatible options
        """
        options = {}

        # Temperature controls randomness in token selection
        if 'temperature' in kwargs:
            options['temperature'] = kwargs['temperature']
            
        # Top-p (nucleus sampling) controls diversity
        if 'top_p' in kwargs:
            options['top_p'] = kwargs['top_p']
            
        # Top-k limits vocabulary considered at each step
        if 'top_k' in kwargs:
            options['top_k'] = kwargs['top_k']
            
        # Map max_tokens to Ollama's num_predict parameter
        if 'max_tokens' in kwargs:
            options['num_predict'] = kwargs['max_tokens']
            
        # Map stop_sequences to Ollama's stop parameter
        if 'stop_sequences' in kwargs:
            options['stop'] = kwargs['stop_sequences']

        return options

    def _stream_response(self, response) -> Iterator[str]:
        """
        Process streaming response from Ollama and yield text chunks.
        
        Ollama streams responses as a series of JSON objects, each containing
        a message delta with incremental content.
        
        Args:
            response: Streaming response iterator from Ollama
            
        Yields:
            String chunks of the generated response as they arrive
        """
        # Iterate through streaming chunks from Ollama
        for chunk in response:
            # Check if chunk contains message content
            if 'message' in chunk and 'content' in chunk['message']:
                # Extract the text content from this chunk
                chunk_text = chunk['message']['content']
                
                # Only yield non-empty chunks
                if chunk_text:
                    yield chunk_text

class LLMClient:
    """Unified LLM client that supports multiple providers"""

    PROVIDERS = {
        'openai': OpenAIProvider,
        'google': GoogleProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider,
    }

    def __init__(self, provider: str = 'google', model: str = None, api_key: str = None):
        self.provider_name = provider.lower()

        if self.provider_name not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.PROVIDERS.keys())}")

        # Check if provider is available
        availability_map = {
            'openai': OPENAI_AVAILABLE,
            'google': GOOGLE_AVAILABLE,
            'anthropic': ANTHROPIC_AVAILABLE,
            'ollama': OLLAMA_AVAILABLE
        }

        if not availability_map[self.provider_name]:
            raise ImportError(f"{provider} provider not available. Please install the required library.")

        # Get API key from environment if not provided
        if api_key is None:
            api_key = self._get_api_key_from_env(provider)

        # Validate API key is available (except for Ollama which doesn't need one)
        if not api_key and self.provider_name != 'ollama':
            available_keys = self._get_env_key_names(provider)
            raise ValueError(f"No API key found for {provider}. Please set one of: {available_keys}")

        # Set default models
        if model is None:
            model = self._get_default_model(provider)

        # Initialize provider
        provider_class = self.PROVIDERS[self.provider_name]
        self.provider = provider_class(api_key=api_key, model=model)

    def _get_api_key_from_env(self, provider: str) -> Optional[str]:
        """
        Get API key from environment variables.
        
        Note: Ollama doesn't require an API key as it runs locally,
        but we return a placeholder to maintain interface consistency.
        """

        env_keys = {
            'openai': ['OPENAI_API_KEY'],
            'google': ['GOOGLE_API_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY'],
            'ollama': [],  # Ollama doesn't require API keys (runs locally)
        }

        for key in env_keys.get(provider.lower(), []):
            api_key = os.getenv(key)
            if api_key:
                return api_key

        return None

    def _get_env_key_names(self, provider: str) -> List[str]:
        """
        Get list of environment variable names for a provider.
        
        Returns:
            List of environment variable names that can contain the API key
        """
        env_keys = {
            'openai': ['OPENAI_API_KEY'],
            'google': ['GOOGLE_API_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY'],
            'ollama': [],  # Ollama doesn't require API keys (runs locally)
        }
        return env_keys.get(provider.lower(), [])

    def _get_default_model(self, provider: str) -> str:
        """Get default model for each provider"""

        return DEFAULT_LLM_MODELS.get(provider)

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """Send chat messages using the configured provider"""
        return self.provider.chat(messages, stream=stream, **kwargs)

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Stream chat response using the configured provider"""
        return self.provider.chat_stream(messages, **kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all available providers"""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """
        List providers that are actually available (libraries installed).
        
        Returns:
            List of provider names that have their required libraries installed
        """
        availability_map = {
            'openai': OPENAI_AVAILABLE,
            'google': GOOGLE_AVAILABLE,
            'anthropic': ANTHROPIC_AVAILABLE,
            'ollama': OLLAMA_AVAILABLE
        }
        return [provider for provider, available in availability_map.items() if available]

    @classmethod
    def check_api_keys(cls) -> Dict[str, bool]:
        """Check which providers have API keys available in environment"""
        results = {}
        temp_client = cls.__new__(cls)  # Create instance without calling __init__

        for provider in cls.list_available_providers():
            api_key = temp_client._get_api_key_from_env(provider)
            results[provider] = bool(api_key)

        return results

# Convenience function for backwards compatibility
def create_llm_client(backend: str = 'google', model: str = None, api_key: str = None) -> LLMClient:
    """Create an LLM client with the specified backend"""
    return LLMClient(provider=backend, model=model, api_key=api_key)