"""
Simple configuration management using environment variables and .env.old files.

This module provides a minimal configuration system that just reads environment
variables and detects available providers. No JSON files, no complex classes.
"""

import os
from typing import Optional, List, Dict, Any

from models.image_data import ProviderConfig




def get_provider_config(provider_name: str) -> Optional[ProviderConfig]:
    """
    Get configuration for a specific provider from environment variables.
    
    Args:
        provider_name: Name of the provider ('ollama', 'anthropic', etc.)
        
    Returns:
        ProviderConfig if provider is configured, None otherwise
    """
    if provider_name == 'ollama':
        settings = {
            'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
            'model': os.getenv('OLLAMA_MODEL', 'llama3.2-vision:latest'),
            'timeout': int(os.getenv('OLLAMA_TIMEOUT', '300'))
        }
        return ProviderConfig(provider_name='ollama', settings=settings)
    
    elif provider_name == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return None
        
        settings = {
            'api_key': api_key,
            'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-7-sonnet-20250219'),
            'max_tokens': int(os.getenv('ANTHROPIC_MAX_TOKENS', '2000')),
            'temperature': float(os.getenv('ANTHROPIC_TEMPERATURE', '0.2'))
        }
        return ProviderConfig(provider_name='anthropic', settings=settings)
    
    elif provider_name == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
        
        settings = {
            'api_key': api_key,
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
            'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.2'))
        }
        return ProviderConfig(provider_name='openai', settings=settings)
    
    elif provider_name == 'bedrock':
        if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')):
            return None
        
        settings = {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
            'model': os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
        }
        return ProviderConfig(provider_name='bedrock', settings=settings)
    
    return None
