"""
LLM Provider implementations for image categorization.

This package contains different LLM provider implementations that can be used
for image description and categorization.
"""

from .base import BaseLLMProvider
from .ollama_provider import OllamaProvider

__all__ = ['BaseLLMProvider', 'OllamaProvider']