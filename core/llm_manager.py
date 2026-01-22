"""
LLM Manager - OpenAI API Integration
Provides API-based LLM access without GPU requirements
"""

from typing import Optional, Dict, Any, List
import openai
from openai import OpenAI
import tiktoken
import time
from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger("llm_manager")


class LLMManager:
    """
    Manages interaction with OpenAI API for reasoning tasks.
    Optimized for GPT-4 with fallback to GPT-3.5-turbo for cost savings.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize LLM Manager
        
        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model to use (defaults to settings, typically gpt-4)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            logger.warning(f"Model {self.model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Context window sizes
        self.context_windows = {
            "gpt-4": 8192,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }
        self.max_tokens = self.context_windows.get(self.model, 8192)
        
        logger.info(f"LLM Manager initialized with model: {self.model}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate API cost for a request
        
        Args:
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (update as needed)
        pricing = {
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-4-turbo-preview": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
            "gpt-3.5-turbo-16k": {"prompt": 0.003 / 1000, "completion": 0.004 / 1000},
        }
        
        model_pricing = pricing.get(self.model, pricing["gpt-4"])
        cost = (prompt_tokens * model_pricing["prompt"] + 
                completion_tokens * model_pricing["completion"])
        return cost
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion using the LLM
        
        Args:
            prompt: User prompt/query
            system_message: System message for context/instructions
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Response format specification (e.g., {"type": "json_object"})
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Dict containing response, token counts, and metadata
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Count input tokens
        input_text = system_message or "" + prompt
        input_tokens = self.count_tokens(input_text)
        
        # Validate context window
        if input_tokens > self.max_tokens - 500:  # Leave room for response
            raise ValueError(
                f"Input too long: {input_tokens} tokens exceeds "
                f"model limit of {self.max_tokens}"
            )
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries})")
                
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": self.timeout,
                }
                
                if max_tokens:
                    api_params["max_tokens"] = max_tokens
                
                if response_format:
                    api_params["response_format"] = response_format
                
                # Merge additional kwargs
                api_params.update(kwargs)
                
                response = self.client.chat.completions.create(**api_params)
                
                # Extract response data
                output = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # Token counts
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Estimate cost
                cost = self.estimate_cost(prompt_tokens, completion_tokens)
                
                logger.info(
                    f"LLM response generated: {completion_tokens} tokens, "
                    f"cost: ${cost:.4f}"
                )
                
                return {
                    "output": output,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "model": self.model,
                }
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise
            
            except Exception as e:
                logger.error(f"Unexpected error in LLM generation: {e}")
                raise
    
    def generate_with_fallback(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        fallback_model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with automatic fallback to cheaper model on error
        
        Args:
            prompt: User prompt
            system_message: System message
            fallback_model: Model to fallback to (default: gpt-3.5-turbo)
            **kwargs: Additional parameters
            
        Returns:
            Response dict with metadata
        """
        try:
            return self.generate(prompt, system_message, **kwargs)
        except Exception as e:
            logger.warning(f"Primary model failed, falling back to {fallback_model}: {e}")
            
            # Temporarily switch model
            original_model = self.model
            self.model = fallback_model
            
            try:
                result = self.generate(prompt, system_message, **kwargs)
                result["fallback_used"] = True
                result["original_model"] = original_model
                return result
            finally:
                self.model = original_model
    
    def batch_generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for multiple prompts
        
        Args:
            prompts: List of prompts
            system_message: System message (shared across all)
            **kwargs: Additional parameters
            
        Returns:
            List of response dicts
        """
        results = []
        total_cost = 0.0
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt, system_message, **kwargs)
            results.append(result)
            total_cost += result.get("cost", 0.0)
        
        logger.info(f"Batch generation complete. Total cost: ${total_cost:.4f}")
        return results


# Singleton instance
_llm_manager_instance: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get or create singleton LLM Manager instance"""
    global _llm_manager_instance
    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()
    return _llm_manager_instance
