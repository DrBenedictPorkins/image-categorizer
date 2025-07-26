"""
Base interface for LLM providers.

This module defines the abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
import time

from models.image_data import CategorizationResult, ProviderConfig


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers that handle image description and categorization."""
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider configuration containing settings
        """
        self.config = config
        self._initialized = False
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of the provider."""
        pass
    
    @property
    @abstractmethod
    def required_config_keys(self) -> List[str]:
        """List of required configuration keys for this provider."""
        pass
    
    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports vision/image input."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Whether the provider has been successfully initialized."""
        return self._initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider with the given configuration.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Prints error and exits if configuration is invalid.
        """
        pass
    
    @abstractmethod
    def process_images(
        self, 
        image_paths: List[str], 
        progress_callback: Optional[Callable[[str, float], None]] = None,
        initial_categories: Optional[List[str]] = None
    ) -> CategorizationResult:
        """
        Process a list of images: describe AND categorize them.
        
        This is the main method that providers must implement. It should:
        1. Generate descriptions for each image
        2. Analyze all images collectively to create meaningful categories
        3. Assign each image to appropriate categories
        4. Return a complete CategorizationResult
        
        Args:
            image_paths: List of absolute paths to image files
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)
            initial_categories: Optional list of initial category suggestions for the LLM
        
        Returns:
            CategorizationResult containing all processed images and categorization
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the provider can connect to its backend service.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get provider capabilities and limitations.
        
        Returns:
            Dictionary describing provider capabilities
        """
        return {
            'provider_name': self.provider_name,
            'supports_vision': self.supports_vision,
            'supports_batch_processing': True,
            'max_image_size': None,  # Override in subclasses if there are limits
            'supported_formats': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            'concurrent_requests': 1  # Override in subclasses for concurrent processing
        }
    
    def cleanup(self):
        """
        Clean up resources used by the provider.
        
        Override in subclasses if cleanup is needed.
        """
        pass
    
    def _report_progress(
        self, 
        callback: Optional[Callable[[str, float], None]], 
        message: str, 
        progress: float
    ):
        """
        Helper method to report progress if callback is provided.
        
        Args:
            callback: Progress callback function
            message: Progress message
            progress: Progress value between 0.0 and 1.0
        """
        if callback:
            callback(message, max(0.0, min(1.0, progress)))
    
    def _validate_image_paths(self, image_paths: List[str]) -> List[str]:
        """
        Validate that image paths exist and are readable.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of valid image paths
        """
        valid_paths = []
        supported_extensions = self.get_capabilities()['supported_formats']
        
        for path in image_paths:
            path_obj = Path(path)
            
            # Check if file exists
            if not path_obj.exists():
                print(f"Warning: Image file does not exist: {path}")
                continue
            
            # Check if it's a file
            if not path_obj.is_file():
                print(f"Warning: Path is not a file: {path}")
                continue
            
            # Check file extension
            extension = path_obj.suffix.lower().lstrip('.')
            if extension not in supported_extensions:
                print(f"Warning: Unsupported image format: {path} (extension: {extension})")
                continue
            
            valid_paths.append(path)
        
        return valid_paths
    
    def _retry_with_exponential_backoff(
        self, 
        func: Callable, 
        max_retries: int = 3, 
        retry_delay: float = 1.0, 
        retry_exceptions: tuple = (Exception,),
        operation_name: str = "operation"
    ):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            retry_exceptions: Tuple of exceptions that should trigger a retry
            operation_name: Human-readable name for logging
            
        Returns:
            Result of the function call
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except retry_exceptions as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    print(f"{operation_name} failed, retrying... (attempt {attempt + 1}/{max_retries}, Error: {e})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"{operation_name} failed after {max_retries} attempts: {e}")
                    break
        
        # All attempts failed
        raise last_exception


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass


class ProviderConnectionError(ProviderError):
    """Exception raised when provider cannot connect to its backend."""
    pass


class ProviderConfigurationError(ProviderError):
    """Exception raised when provider configuration is invalid."""
    pass


class ProviderProcessingError(ProviderError):
    """Exception raised during image processing."""
    pass