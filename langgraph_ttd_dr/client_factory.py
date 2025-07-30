import os
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_openai_client(
    use_azure: Optional[bool] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create an OpenAI or Azure OpenAI client
    
    Args:
        use_azure: Force Azure OpenAI (True) or OpenAI (False), auto-detect if None
        api_key: API key for authentication
        endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        **kwargs: Additional arguments
        
    Returns:
        Configured client
    """
    
    # Auto-detect if not specified
    if use_azure is None:
        use_azure = bool(os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_OPENAI_API_KEY'))
    
    if use_azure:
        return _create_azure_client(api_key, endpoint, api_version, **kwargs)
    else:
        return _create_openai_client(api_key, **kwargs)


def _create_azure_client(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs
) -> Any:
    """Create Azure OpenAI client"""
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.error("Azure OpenAI requires openai>=1.0.0")
        raise ImportError("Please install openai>=1.0.0")
    
    azure_endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
    azure_api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    
    if not azure_endpoint:
        raise ValueError("Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT")
    if not azure_api_key:
        raise ValueError("Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY")
    
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
        **kwargs
    )
    
    logger.info(f"Azure OpenAI client created - {azure_endpoint}")
    return client


def _create_openai_client(api_key: Optional[str] = None, **kwargs) -> Any:
    """Create regular OpenAI client"""
    from openai import OpenAI
    
    openai_api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY")
    
    client = OpenAI(api_key=openai_api_key, **kwargs)
    logger.info("OpenAI client created")
    return client


def get_model_name(use_azure: Optional[bool] = None) -> str:
    """Get appropriate model name"""
    if use_azure is None:
        use_azure = bool(os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_OPENAI_API_KEY'))
    
    if use_azure:
        return (os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME') or 
                os.getenv('AZURE_OPENAI_CHATGPT_DEPLOYMENT') or 
                "gpt-4.1-nano")
    else:
        return os.getenv('OPENAI_MODEL') or "gpt-4.1-nano"


def detect_config():
    """Detect available configuration"""
    has_azure = bool(os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_OPENAI_API_KEY'))
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    
    return {
        "has_azure": has_azure,
        "has_openai": has_openai,
        "recommended": "azure" if has_azure else "openai" if has_openai else None
    } 