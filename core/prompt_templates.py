"""
Prompt Templates Module
Manages system prompt templates for different tagging styles.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from .config_manager import get_config_dir


# Default prompt templates - simplified to 2 main types
DEFAULT_TEMPLATES = {
    "Danbooru Tag": {
        "description": "Danbooru-style tags for LoRA/Fine-tuning training",
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
1girl, solo, short hair, red hair, green eyes, hoodie, black pants, hands in pockets, standing, white background, simple background, looking at viewer, expressionless"""
    },
    "Natural Caption": {
        "description": "Detailed natural language description for datasets",
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
"A digital illustration features a young woman with disheveled silver hair and piercing red eyes, standing in a dimly lit, industrial corridor. She wears a tattered black gothic dress with white lace trim that is stained with patches of red. Her posture is aggressive; she leans forward with a manic expression, gripping a serrated silver knife in her right hand, positioned as if ready to strike. The lighting is low-key, with a cool blue hue casting long, dramatic shadows against the rusted metal walls behind her. The art style utilizes high contrast and sharp line work to emphasize a tense, horror-inspired atmosphere.\""""
    },
}


def get_templates_path() -> Path:
    """Get the templates file path."""
    return get_config_dir() / 'prompt_templates.json'


class PromptTemplateManager:
    """Manages system prompt templates."""
    
    def __init__(self):
        self._templates: Dict[str, dict] = {}
        self._templates_path = get_templates_path()
        self.load()
    
    def load(self) -> bool:
        """Load templates from file, merging with defaults."""
        # Start with defaults
        self._templates = {k: v.copy() for k, v in DEFAULT_TEMPLATES.items()}
        
        try:
            if self._templates_path.exists():
                with open(self._templates_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    # Merge saved templates (user templates override defaults)
                    self._templates.update(saved)
                return True
        except Exception as e:
            print(f"Error loading templates: {e}")
        return False
    
    def save(self) -> bool:
        """Save templates to file."""
        try:
            # Only save non-default templates or modified defaults
            to_save = {}
            for name, template in self._templates.items():
                if name not in DEFAULT_TEMPLATES or template != DEFAULT_TEMPLATES[name]:
                    to_save[name] = template
            
            with open(self._templates_path, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
        return False
    
    def get_names(self) -> List[str]:
        """Get list of template names."""
        return list(self._templates.keys())
    
    def get(self, name: str) -> Optional[dict]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def get_prompt(self, name: str) -> str:
        """Get the prompt text for a template."""
        template = self._templates.get(name)
        if template:
            return template.get('prompt', '')
        return ''
    
    def get_format(self, name: str) -> str:
        """Get the format type for a template (tag/captioning)."""
        template = self._templates.get(name)
        if template:
            return template.get('format', 'captioning')
        return 'captioning'
    
    def add(self, name: str, prompt: str, format_type: str = "tag", description: str = "") -> bool:
        """Add or update a template."""
        if not name or not prompt:
            return False
        
        self._templates[name] = {
            "description": description,
            "format": format_type,
            "prompt": prompt
        }
        return self.save()
    
    def delete(self, name: str) -> bool:
        """Delete a template."""
        if name in self._templates:
            del self._templates[name]
            return self.save()
        return False
    
    def is_default(self, name: str) -> bool:
        """Check if a template is a default template."""
        return name in DEFAULT_TEMPLATES


# Global instance
_template_manager: Optional[PromptTemplateManager] = None


def get_templates() -> PromptTemplateManager:
    """Get or create the global PromptTemplateManager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager
