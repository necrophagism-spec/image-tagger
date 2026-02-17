"""
Gemini API Module
Handles inference using Google's Gemini API.
"""

from typing import Optional, List
from PIL import Image
import base64
from io import BytesIO

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


# Available Gemini models for vision (updated 2026-02)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.0-flash",
]

# Map our unified reasoning effort levels to Gemini thinking_level (Gemini 3)
_EFFORT_TO_THINKING_LEVEL = {
    "none": "minimal",     # Gemini 3 can't fully disable; minimal = likely no thinking
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "auto": None,          # Let model decide dynamically
}

# Map our unified reasoning effort levels to Gemini thinking_budget (Gemini 2.5)
_EFFORT_TO_THINKING_BUDGET = {
    "none": 0,        # Disable thinking (not supported by 2.5 Pro, will be ignored)
    "minimal": 256,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "auto": -1,       # Dynamic allocation
}


class GeminiAPI:
    """Gemini API inference for image tagging."""
    
    def __init__(self):
        self.client: Optional[genai.Client] = None
        self.api_key: Optional[str] = None
        self.model_name: str = "gemini-2.5-flash"
    
    @staticmethod
    def is_available() -> bool:
        """Check if google-genai is installed."""
        return GENAI_AVAILABLE
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available Gemini vision models."""
        return GEMINI_MODELS.copy()
    
    def configure(self, api_key: str, model_name: str = "gemini-2.5-flash") -> bool:
        """
        Configure the Gemini API client.
        
        Args:
            api_key: Google AI Studio API key
            model_name: Model to use for generation
            
        Returns:
            True if successful
        """
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-genai is not installed")
        
        try:
            self.client = genai.Client(api_key=api_key)
            self.api_key = api_key
            self.model_name = model_name
            return True
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if API is configured."""
        return self.client is not None
    
    def _image_to_part(self, image: Image.Image) -> dict:
        """Convert PIL Image to Gemini Part format."""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        return types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )
    
    def _build_thinking_config(self, reasoning_effort: str):
        """Build ThinkingConfig based on model and effort level."""
        if not reasoning_effort or reasoning_effort == "auto":
            return None  # Let model decide
        
        is_gemini3 = "gemini-3" in self.model_name
        
        if is_gemini3:
            level = _EFFORT_TO_THINKING_LEVEL.get(reasoning_effort)
            if level:
                return types.ThinkingConfig(thinking_level=level)
        else:
            budget = _EFFORT_TO_THINKING_BUDGET.get(reasoning_effort, -1)
            return types.ThinkingConfig(thinking_budget=budget)
        
        return None
    
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
            top_k: Top-K sampling parameter
            top_p: Top-P (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/auto)
            
        Returns:
            Generated text
        """
        if not self.is_configured():
            raise RuntimeError("Gemini API not configured")
        
        # Convert image to part
        image_part = self._image_to_part(image)
        
        # Build thinking config
        thinking_config = self._build_thinking_config(reasoning_effort)
        
        # Build generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None,
            thinking_config=thinking_config,
        )
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_part, user_prompt],
            config=config,
        )
        
        return response.text


# Global instance for easy access
_gemini_api: Optional[GeminiAPI] = None


def get_gemini_api() -> GeminiAPI:
    """Get or create the global GeminiAPI instance."""
    global _gemini_api
    if _gemini_api is None:
        _gemini_api = GeminiAPI()
    return _gemini_api
