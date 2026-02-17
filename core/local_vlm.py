"""
Local VLM Inference Module
Handles inference using llama-cpp-python with multimodal support.
Supports both LLaVA and Qwen3VL model architectures.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import base64
from io import BytesIO


class VLMType(Enum):
    """Supported VLM model types."""
    LLAVA = "llava"
    QWEN3VL = "qwen3vl"


# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_AVAILABLE = True
    LLAVA_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    LLAVA_AVAILABLE = False
    Llama = None
    Llava15ChatHandler = None

# Try to import Qwen3VL handler (JamePeng's fork)
try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    Qwen3VLChatHandler = None


class LocalVLM:
    """Local VLM inference using llama-cpp-python."""
    
    def __init__(self):
        self.model: Optional[Llama] = None
        self.chat_handler = None
        self.model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.model_type: VLMType = VLMType.QWEN3VL
    
    @staticmethod
    def is_available() -> bool:
        """Check if llama-cpp-python is installed."""
        return LLAMA_CPP_AVAILABLE
    
    @staticmethod
    def is_qwen3vl_available() -> bool:
        """Check if Qwen3VL support is available (JamePeng's fork)."""
        return QWEN3VL_AVAILABLE
    
    @staticmethod
    def get_available_types() -> list:
        """Get list of available VLM types."""
        types = []
        if LLAVA_AVAILABLE:
            types.append(VLMType.LLAVA)
        if QWEN3VL_AVAILABLE:
            types.append(VLMType.QWEN3VL)
        return types
    
    def load_model(
        self,
        model_path: str,
        mmproj_path: str,
        model_type: VLMType = VLMType.QWEN3VL,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        force_reasoning: bool = False
    ) -> bool:
        """
        Load a VLM model with multimodal projector.
        
        Args:
            model_path: Path to the .gguf model file
            mmproj_path: Path to the mmproj.gguf file
            model_type: Type of VLM model (LLAVA or QWEN3VL)
            n_ctx: Context window size (default 8192 for Qwen3VL)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            force_reasoning: Enable thinking mode for Qwen3VL-Thinking models
            
        Returns:
            True if successful, False otherwise
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not installed")
        
        try:
            self.model_type = model_type
            
            # Create appropriate chat handler based on model type
            if model_type == VLMType.QWEN3VL:
                if not QWEN3VL_AVAILABLE:
                    raise RuntimeError(
                        "Qwen3VL support not available. "
                        "Please install JamePeng's llama-cpp-python fork from: "
                        "https://github.com/JamePeng/llama-cpp-python/releases/"
                    )
                
                self.chat_handler = Qwen3VLChatHandler(
                    clip_model_path=mmproj_path,
                    force_reasoning=force_reasoning,
                    image_min_tokens=1024,  # Required for Qwen3VL
                )
                
                # Qwen3VL needs larger context and swa_full
                self.model = Llama(
                    model_path=model_path,
                    chat_handler=self.chat_handler,
                    n_ctx=max(n_ctx, 8192),
                    n_gpu_layers=n_gpu_layers,
                    swa_full=True,  # Required for Qwen3VL
                )
            else:
                # LLaVA models
                if not LLAVA_AVAILABLE:
                    raise RuntimeError("LLaVA support not available")
                
                self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                
                self.model = Llama(
                    model_path=model_path,
                    chat_handler=self.chat_handler,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    logits_all=True,
                )
            
            self.model_path = model_path
            self.mmproj_path = mmproj_path
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.chat_handler = None
            return False
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        self.model = None
        self.chat_handler = None
        self.model_path = None
        self.mmproj_path = None
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to data URI for llama.cpp."""
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
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        max_tokens: int = 512
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
            min_p: Min-P sampling parameter
            repeat_penalty: Repetition penalty
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        # Convert image to data URI
        image_uri = self._image_to_data_uri(image)
        
        # Build messages with image
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Generate response
        response = self.model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
        )
        
        return response['choices'][0]['message']['content']


# Global instance for easy access
_local_vlm: Optional[LocalVLM] = None


def get_local_vlm() -> LocalVLM:
    """Get or create the global LocalVLM instance."""
    global _local_vlm
    if _local_vlm is None:
        _local_vlm = LocalVLM()
    return _local_vlm
