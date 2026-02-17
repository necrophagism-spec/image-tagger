"""
Unified Tagger Module
Provides a unified interface for both local VLM and Gemini API backends.
"""

from enum import Enum
from typing import Optional, Callable
from pathlib import Path
from PIL import Image

from .local_vlm import get_local_vlm, LocalVLM
from .gemini_api import get_gemini_api, GeminiAPI
from .openai_compatible_api import get_xai_api, get_openrouter_api, OpenAICompatibleAPI
from .image_processor import find_images, load_image, save_tags


class TaggingFormat(Enum):
    """Tagging output format."""
    CAPTIONING = "captioning"
    TAG = "tag"


class BackendType(Enum):
    """Inference backend type."""
    LOCAL_VLM = "local"
    GEMINI_API = "gemini"
    XAI = "xai"
    OPENROUTER = "openrouter"


# Default system prompt (used as fallback)
DEFAULT_SYSTEM_PROMPT = "You are an expert image tagger for anime, illustrations, and photographs."


class Tagger:
    """Unified interface for image tagging."""
    
    def __init__(self):
        self.backend_type: BackendType = BackendType.GEMINI_API
        self.format: TaggingFormat = TaggingFormat.CAPTIONING
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        
        # Generation parameters
        self.temperature: float = 0.4
        self.top_k: int = 40
        self.top_p: float = 0.9
        self.min_p: float = 0.05
        self.repeat_penalty: float = 1.1
        self.max_tokens: int = 512
        
        # Callbacks
        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None
        self.on_complete: Optional[Callable[[int], None]] = None
        
        # Control
        self._stop_requested: bool = False
    
    def stop(self) -> None:
        """Request to stop processing."""
        self._stop_requested = True
    
    def _get_backend(self):
        """Get the appropriate backend instance."""
        if self.backend_type == BackendType.LOCAL_VLM:
            return get_local_vlm()
        elif self.backend_type == BackendType.XAI:
            return get_xai_api()
        elif self.backend_type == BackendType.OPENROUTER:
            return get_openrouter_api()
        else:
            return get_gemini_api()
    
    def _generate(self, image: Image.Image) -> str:
        """Generate tags for a single image."""
        backend = self._get_backend()
        
        # Use system_prompt as the main instruction
        # The user_prompt is a simple trigger to analyze the image
        user_prompt = "Analyze this image and follow the instructions provided."
        
        if self.backend_type == BackendType.LOCAL_VLM:
            return backend.generate(
                image=image,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                repeat_penalty=self.repeat_penalty,
                max_tokens=self.max_tokens,
            )
        else:
            # Gemini, xAI, and OpenRouter all share the same parameter interface
            return backend.generate(
                image=image,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
    
    def process_folder(self, folder_path: str) -> int:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to the folder containing images
            
        Returns:
            Number of successfully processed images
        """
        self._stop_requested = False
        
        # Find all images
        images = find_images(folder_path)
        total = len(images)
        
        if total == 0:
            if self.on_complete:
                self.on_complete(0)
            return 0
        
        # Check backend is ready
        backend = self._get_backend()
        if self.backend_type == BackendType.LOCAL_VLM:
            if not backend.is_loaded():
                if self.on_error:
                    self.on_error("", "No local model loaded")
                return 0
        else:
            if not backend.is_configured():
                if self.on_error:
                    self.on_error("", "Gemini API not configured")
                return 0
        
        # Process each image
        processed = 0
        for i, image_path in enumerate(images):
            if self._stop_requested:
                break
            
            filename = image_path.name
            
            # Report progress
            if self.on_progress:
                self.on_progress(i + 1, total, filename)
            
            try:
                # Load image
                image = load_image(image_path)
                
                # Generate tags
                tags = self._generate(image)
                
                # Save output
                save_tags(image_path, tags.strip())
                processed += 1
                
            except Exception as e:
                if self.on_error:
                    self.on_error(filename, str(e))
        
        # Report completion
        if self.on_complete:
            self.on_complete(processed)
        
        return processed
    
    def process_images(
        self, 
        image_paths: list, 
        on_image_done: Optional[Callable[[Path], None]] = None
    ) -> int:
        """
        Process a specific list of images.
        
        Args:
            image_paths: List of Path objects for images to process
            on_image_done: Callback called after each image is processed
            
        Returns:
            Number of successfully processed images
        """
        self._stop_requested = False
        total = len(image_paths)
        
        if total == 0:
            if self.on_complete:
                self.on_complete(0)
            return 0
        
        # Check backend is ready
        backend = self._get_backend()
        if self.backend_type == BackendType.LOCAL_VLM:
            if not backend.is_loaded():
                if self.on_error:
                    self.on_error("", "No local model loaded")
                return 0
        else:
            if not backend.is_configured():
                if self.on_error:
                    self.on_error("", "Gemini API not configured")
                return 0
        
        # Process each image
        processed = 0
        for i, image_path in enumerate(image_paths):
            if self._stop_requested:
                break
            
            filename = image_path.name
            
            # Report progress
            if self.on_progress:
                self.on_progress(i + 1, total, filename)
            
            try:
                # Load image
                image = load_image(image_path)
                
                # Generate tags
                tags = self._generate(image)
                
                # Save output
                save_tags(image_path, tags.strip())
                processed += 1
                
                # Call per-image callback
                if on_image_done:
                    on_image_done(image_path)
                
            except Exception as e:
                if self.on_error:
                    self.on_error(filename, str(e))
        
        # Report completion
        if self.on_complete:
            self.on_complete(processed)
        
        return processed


# Global instance
_tagger: Optional[Tagger] = None


def get_tagger() -> Tagger:
    """Get or create the global Tagger instance."""
    global _tagger
    if _tagger is None:
        _tagger = Tagger()
    return _tagger
