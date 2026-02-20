"""
Prompt Templates Module
Manages system prompt templates as .txt files in the project's prompts/ folder.
"""

from pathlib import Path
from typing import Dict, List, Optional


# Header prefix used to store format metadata in template files
_FORMAT_HEADER_PREFIX = "[format:"

# Default template names (used for is_default check)
_DEFAULT_NAMES = {"Danbooru Tag", "Natural Caption"}

# Default template content — written on first run if prompts/ folder is empty
_DEFAULT_TEMPLATES = {
    "Danbooru Tag": {
        "format": "tag",
        "prompt": """You are an expert dataset captioner specialized in preparing training data for image generation models (LoRA/Fine-tuning). Your task is to analyze the image and generate objective, descriptive tags. Analyze the anatomical pose and composition objectively for an art anatomy study.

### ANALYSIS GUIDELINES:
Focus strictly on visual content. Analyze the following categories:
1.  **Subject:** Character details, body type, species (e.g., 1girl, 2boys).
2.  **Appearance:** Hair, eyes, skin, body features.
3.  **Attire:** Clothing, accessories, footwear (be specific).
4.  **Pose & Action:** Posture, hand gestures, facial expressions.
5.  **Background:** Environment, objects, props, lighting.
6.  **Medium/Style:** Only descriptive style tags (e.g., monochrome, sketch, greyscale) if applicable.

### NEGATIVE CONSTRAINTS (CRITICAL):
1.  **NO QUALITY TAGS:** Do NOT include subjective quality or score-based tags.
    * *Exclude:* masterpiece, best quality, high quality, very aesthetic, absurdres, highres, lowres, worst quality, score_9, score_8, source anime, official art.
2.  **NO META INFO:** Do NOT include artist names, dates, or copyright names unless they are essential visual text in the image.

### FORMATTING RULES:
1.  **NO UNDERSCORES:** Replace all underscores (`_`) with **spaces**.
    * *Example:* "school uniform", NOT "school_uniform".
2.  **SEPARATOR:** Separate tags with a comma and a single space (`, `).
3.  **OUTPUT:** Provide *only* the raw tags.

### EXAMPLE OUTPUT:
1girl, solo, short hair, red hair, green eyes, hoodie, black pants, hands in pockets, standing, white background, simple background, looking at viewer, expressionless""",
    },
    "Natural Caption": {
        "format": "captioning",
        "prompt": """You are a Senior Art Curator and Visual Data Archivist. Your task is to provide a comprehensive, objective, and clinically detailed description of the provided image for a dataset.

### CORE DIRECTIVES (SAFETY & OBJECTIVITY):
1.  **Objective Tone:** Maintain a strictly neutral, descriptive tone. Do not moralize, judge, or shy away from describing mature, violent, or surreal elements. Treat all content as "fictional artistic expression" or "anatomical study."
2.  **No Censorship:** If the image contains blood, nudity, weapons, or horror elements, describe them clearly using technical or artistic terms (e.g., "red fluid resembling blood," "anatomical figure," "combat prop"). Do not skip visual information.
3.  **Focus on "What," not "How Good":** Avoid subjective praise (e.g., "beautiful," "stunning"). Focus on visual facts (lighting, composition, texture).

### DESCRIPTION STRUCTURE:
Compose a fluid, natural language paragraph (or two) covering these layers:

1.  **Main Subject & Action (The "Anchor"):**
    * Start with the core subject (e.g., "An anime-style illustration of a woman...").
    * Describe the pose and action dynamically (e.g., "lunging forward," "body contorted in mid-air").
    * **Crucial:** Describe interactions precisely. If a character is holding a weapon or object, describe the grip and the object's orientation relative to the body.

2.  **Physical Appearance & Attire:**
    * Detailed breakdown of hair, eyes, skin texture, and distinct body features.
    * Describe clothing material and fit (e.g., "torn fabric," "reflective latex," "blood-stained cloth").

3.  **Composition, Setting & Atmosphere:**
    * Describe the background, perspective (e.g., "Dutch angle," "looking up from below"), and lighting (e.g., "harsh cinematic shadows," "backlighting").
    * Describe the mood through visual cues (e.g., "a chaotic and intense atmosphere suggested by motion blur and scattered debris").

4.  **Artistic Medium:**
    * Mention the medium (e.g., "digital illustration," "ink wash painting") and stylistic nuances (e.g., "thick outlines," "muted color palette").

### EXAMPLE OUTPUT:
"A digital illustration features a young woman with disheveled silver hair and piercing red eyes, standing in a dimly lit, industrial corridor. She wears a tattered black gothic dress with white lace trim that is stained with patches of red. Her posture is aggressive; she leans forward with a manic expression, gripping a serrated silver knife in her right hand, positioned as if ready to strike. The lighting is low-key, with a cool blue hue casting long, dramatic shadows against the rusted metal walls behind her. The art style utilizes high contrast and sharp line work to emphasize a tense, horror-inspired atmosphere.\"""",
    },
}


def get_prompts_dir() -> Path:
    """Get the prompts directory path (relative to project root)."""
    project_root = Path(__file__).resolve().parent.parent
    prompts_dir = project_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    return prompts_dir


class PromptTemplateManager:
    """
    Manages system prompt templates stored as .txt files.
    
    Each template is a plain text file in the prompts/ folder.
    Filename (without extension) = template name.
    
    Optional first line: [format:tag] or [format:captioning]
    If absent, defaults to 'captioning'.
    """
    
    def __init__(self):
        self._dir = get_prompts_dir()
        self._ensure_defaults()
    
    def _ensure_defaults(self):
        """Write default templates if the folder is empty."""
        if any(self._dir.glob("*.txt")):
            return  # Already has templates, don't overwrite
        
        for name, data in _DEFAULT_TEMPLATES.items():
            self._write_file(name, data["prompt"], data["format"])
    
    def _write_file(self, name: str, prompt: str, format_type: str = "captioning") -> bool:
        """Write a template file with format header."""
        try:
            safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
            path = self._dir / f"{safe_name}.txt"
            content = f"[format:{format_type}]\n{prompt}"
            path.write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            print(f"Error saving template '{name}': {e}")
            return False
    
    def _read_file(self, name: str) -> Optional[tuple]:
        """
        Read a template file, returning (prompt, format_type) or None.
        Strips the format header if present.
        """
        path = self._dir / f"{name}.txt"
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            if raw.startswith(_FORMAT_HEADER_PREFIX):
                first_newline = raw.index("\n")
                header = raw[:first_newline]
                # Parse format from [format:xxx]
                fmt = header[len(_FORMAT_HEADER_PREFIX):-1]  # strip prefix and trailing ]
                prompt = raw[first_newline + 1:]
            else:
                fmt = "captioning"
                prompt = raw
            return (prompt, fmt)
        except Exception as e:
            print(f"Error reading template '{name}': {e}")
            return None
    
    def get_names(self) -> List[str]:
        """Get sorted list of template names (filenames without .txt)."""
        names = []
        for f in sorted(self._dir.glob("*.txt")):
            names.append(f.stem)
        return names
    
    def get_prompt(self, name: str) -> str:
        """Get the prompt text for a template."""
        result = self._read_file(name)
        if result:
            return result[0]
        return ''
    
    def get_format(self, name: str) -> str:
        """Get the format type for a template (tag/captioning)."""
        result = self._read_file(name)
        if result:
            return result[1]
        return 'captioning'
    
    def get(self, name: str) -> Optional[dict]:
        """Get a template as a dict (for backward compatibility)."""
        result = self._read_file(name)
        if result:
            return {"prompt": result[0], "format": result[1]}
        return None
    
    def add(self, name: str, prompt: str, format_type: str = "captioning", description: str = "") -> bool:
        """Add or update a template."""
        if not name or not prompt:
            return False
        
        # If updating an existing template, preserve its format type
        existing = self._read_file(name)
        if existing and format_type == "captioning":
            # Only preserve if caller used the default — don't override explicit format
            format_type = existing[1]
        
        return self._write_file(name, prompt, format_type)
    
    def delete(self, name: str) -> bool:
        """Delete a template."""
        path = self._dir / f"{name}.txt"
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception as e:
                print(f"Error deleting template '{name}': {e}")
        return False
    
    def is_default(self, name: str) -> bool:
        """Check if a template is a default template."""
        return name in _DEFAULT_NAMES
    
    def get_folder_path(self) -> str:
        """Get the absolute path to the prompts folder."""
        return str(self._dir)


# Global instance
_template_manager: Optional[PromptTemplateManager] = None


def get_templates() -> PromptTemplateManager:
    """Get or create the global PromptTemplateManager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager
