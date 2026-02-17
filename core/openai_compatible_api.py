"""
OpenAI-Compatible API Module
Handles inference using OpenAI-compatible APIs (xAI Grok, OpenRouter, etc.).
Both xAI and OpenRouter use the OpenAI chat completion format.
"""

from typing import Optional, List
from PIL import Image
import base64
from io import BytesIO

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    OpenAI = None


# ===== xAI Grok Models (Vision-capable) =====
XAI_MODELS = [
    "grok-4",
    "grok-4-fast",
    "grok-4.1",
    "grok-4.1-fast",
    "grok-3",
]

XAI_BASE_URL = "https://api.x.ai/v1"

# ===== OpenRouter Models (Vision-capable, curated) =====
OPENROUTER_MODELS = [
    "qwen/qwen-2.5-vl-72b-instruct",
    "x-ai/grok-4",
    "mistralai/pixtral-large-latest",
    "google/gemini-2.5-flash",
    "openai/gpt-4o",
    "meta-llama/llama-4-scout",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatibleAPI:
    """API inference using OpenAI-compatible endpoints (xAI, OpenRouter, etc.)."""
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = None
        self.model_name: str = ""
    
    @staticmethod
    def is_available() -> bool:
        """Check if openai SDK is installed."""
        return OPENAI_SDK_AVAILABLE
    
    def configure(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
    ) -> bool:
        """
        Configure the API client.
        
        Args:
            api_key: API key for the service
            model_name: Model identifier
            base_url: Base URL for the API endpoint
            
        Returns:
            True if successful
        """
        if not OPENAI_SDK_AVAILABLE:
            raise RuntimeError(
                "openai package is not installed. "
                "Run: pip install openai"
            )
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self.api_key = api_key
            self.base_url = base_url
            self.model_name = model_name
            return True
        except Exception as e:
            print(f"Error configuring API: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if API is configured."""
        return self.client is not None
    
    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to data URI."""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    
    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        max_tokens: int = 512,
        reasoning_effort: str = "none",
    ) -> str:
        """
        Generate text description for an image.
        
        Args:
            image: PIL Image to analyze
            system_prompt: System prompt for the model
            user_prompt: User prompt with tagging instructions
            temperature: Sampling temperature
            top_k: Top-K sampling parameter (not used by OpenAI API)
            top_p: Top-P (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/auto)
            
        Returns:
            Generated text
        """
        if not self.is_configured():
            raise RuntimeError("API not configured")
        
        # Convert image to data URI
        image_uri = self._image_to_data_uri(image)
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        })
        
        # Build extra_body for reasoning control
        extra_body = None
        if reasoning_effort and reasoning_effort != "auto":
            extra_body = {
                "reasoning": {
                    "effort": reasoning_effort,
                    "exclude": True,  # Use reasoning internally but don't include in output
                }
            }
        
        # Generate response
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        response = self.client.chat.completions.create(**kwargs)
        
        return response.choices[0].message.content


# ===== Global instances =====

_xai_api: Optional[OpenAICompatibleAPI] = None
_openrouter_api: Optional[OpenAICompatibleAPI] = None


def get_xai_api() -> OpenAICompatibleAPI:
    """Get or create the global xAI API instance."""
    global _xai_api
    if _xai_api is None:
        _xai_api = OpenAICompatibleAPI()
    return _xai_api


def get_openrouter_api() -> OpenAICompatibleAPI:
    """Get or create the global OpenRouter API instance."""
    global _openrouter_api
    if _openrouter_api is None:
        _openrouter_api = OpenAICompatibleAPI()
    return _openrouter_api
