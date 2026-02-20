"""
Image Processor Module
Handles image file discovery and output file management.
"""

import os
from pathlib import Path
from typing import List, Tuple, Generator
from PIL import Image
import base64
from io import BytesIO


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def find_images(folder_path: str) -> List[Path]:
    """
    Find all supported image files in the target folder.
    
    Args:
        folder_path: Path to the folder to scan
        
    Returns:
        List of Path objects for found images
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    
    images = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(file)
    
    return sorted(images)


def load_image(image_path: Path) -> Image.Image:
    """
    Load an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    return Image.open(image_path).convert('RGB')


def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Output format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_output_path(image_path: Path, output_dir: str = None) -> Path:
    """
    Get the corresponding .txt output path for an image.
    
    Args:
        image_path: Path to the image file
        output_dir: Optional custom directory to save the .txt file
        
    Returns:
        Path to the output .txt file
    """
    if output_dir:
        # Save in the custom output directory but keep the identical filename
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        return out_dir_path / image_path.with_suffix('.txt').name
    
    # Default: save alongside the image
    return image_path.with_suffix('.txt')


def save_tags(image_path: Path, tags: str, output_dir: str = None) -> None:
    """
    Save tagging result to a .txt file.
    
    Args:
        image_path: Path to the original image
        tags: The tagging result text
        output_dir: Optional custom directory for the .txt file
    """
    output_path = get_output_path(image_path, output_dir)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tags)


def process_images_generator(
    folder_path: str
) -> Generator[Tuple[Path, Image.Image], None, None]:
    """
    Generator that yields image paths and loaded images.
    
    Args:
        folder_path: Path to the folder containing images
        
    Yields:
        Tuple of (image_path, PIL Image)
    """
    images = find_images(folder_path)
    for image_path in images:
        try:
            image = load_image(image_path)
            yield image_path, image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
