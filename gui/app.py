"""
GUI Application Module
Enhanced CustomTkinter-based interface with image browser and text editor.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image, ImageTk

from core.tagger import get_tagger, BackendType, TaggingFormat
from core.local_vlm import get_local_vlm, VLMType
from core.gemini_api import get_gemini_api, GEMINI_MODELS
from core.openai_compatible_api import (
    get_xai_api, get_openrouter_api,
    XAI_MODELS, XAI_BASE_URL,
    OPENROUTER_MODELS, OPENROUTER_BASE_URL,
)
from core.config_manager import get_config
from core.prompt_templates import get_templates
from core.image_processor import find_images, get_output_path

# Model type display names and values
MODEL_TYPE_OPTIONS = {
    "Local VLM (.gguf)": "local",
    "Gemini API": "gemini",
    "xAI (Grok)": "xai",
    "OpenRouter": "openrouter",
}
MODEL_TYPE_DISPLAY = {v: k for k, v in MODEL_TYPE_OPTIONS.items()}


# Theme configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Thumbnail size
THUMB_SIZE = (80, 80)


class ImageThumbnail(ctk.CTkFrame):
    """Single image thumbnail with checkbox."""
    
    def __init__(self, parent, image_path: Path, on_select=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.image_path = image_path
        self.on_select = on_select
        self._selected = False
        self._checked = True
        
        # Check if has tags
        self.txt_path = get_output_path(image_path)
        self.has_tags = self.txt_path.exists()
        
        # Layout
        self.configure(fg_color="transparent")
        
        # Checkbox
        self.checkbox_var = ctk.BooleanVar(value=True)
        self.checkbox = ctk.CTkCheckBox(
            self, text="", variable=self.checkbox_var,
            width=20, checkbox_width=18, checkbox_height=18
        )
        self.checkbox.grid(row=0, column=0, padx=(5, 2), pady=5)
        
        # Thumbnail image
        self.thumb_label = ctk.CTkLabel(self, text="", width=THUMB_SIZE[0], height=THUMB_SIZE[1])
        self.thumb_label.grid(row=0, column=1, padx=2, pady=5)
        self.thumb_label.bind("<Button-1>", self._on_click)
        
        # Load thumbnail in background
        self._load_thumbnail()
        
        # Filename
        name = image_path.name
        if len(name) > 15:
            name = name[:12] + "..."
        
        # Status indicator
        status = "✓" if self.has_tags else "○"
        status_color = "green" if self.has_tags else "gray"
        
        self.name_label = ctk.CTkLabel(
            self, text=f"{status} {name}", 
            font=("", 11),
            anchor="w",
            width=120
        )
        self.name_label.grid(row=0, column=2, padx=(2, 5), pady=5, sticky="w")
        self.name_label.bind("<Button-1>", self._on_click)
    
    def _load_thumbnail(self):
        """Load and display thumbnail."""
        try:
            img = Image.open(self.image_path)
            img.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to CTkImage
            self.ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.thumb_label.configure(image=self.ctk_image)
        except Exception as e:
            self.thumb_label.configure(text="ERR")
    
    def _on_click(self, event=None):
        """Handle click to select this thumbnail."""
        if self.on_select:
            self.on_select(self)
    
    def set_selected(self, selected: bool):
        """Set visual selection state."""
        self._selected = selected
        if selected:
            self.configure(fg_color=("gray75", "gray30"))
        else:
            self.configure(fg_color="transparent")
    
    def is_checked(self) -> bool:
        """Check if this image is checked for tagging."""
        return self.checkbox_var.get()
    
    def update_status(self):
        """Update the tag status indicator."""
        self.has_tags = self.txt_path.exists()
        status = "✓" if self.has_tags else "○"
        name = self.image_path.name
        if len(name) > 15:
            name = name[:12] + "..."
        self.name_label.configure(text=f"{status} {name}")


class ImageTaggerApp(ctk.CTk):
    """Main application window with enhanced layout."""
    
    def __init__(self):
        super().__init__()
        
        # Load config
        self.config = get_config()
        self.templates = get_templates()
        
        # Window setup
        self.title("Image Tagger")
        geometry = self.config.get("window_geometry", "1400x900")
        self.geometry(geometry)
        self.minsize(1200, 700)
        
        # State
        self._processing = False
        self._process_thread: Optional[threading.Thread] = None
        self._thumbnails: List[ImageThumbnail] = []
        self._selected_thumbnail: Optional[ImageThumbnail] = None
        self._current_images: List[Path] = []
        
        # Protocol for window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Build UI
        self._create_layout()
        self._load_settings()
        self._setup_callbacks()
    
    def _create_layout(self):
        """Create the main 3-column layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=0, minsize=250)  # Thumbnails
        self.grid_columnconfigure(1, weight=1, minsize=400)  # Settings
        self.grid_columnconfigure(2, weight=1, minsize=350)  # Editor
        self.grid_rowconfigure(1, weight=1)
        
        # ===== Top Bar (Folder Selection) =====
        self._create_top_bar()
        
        # ===== Left Column (Image Browser) =====
        self._create_image_browser()
        
        # ===== Center Column (Settings) =====
        self._create_settings_panel()
        
        # ===== Right Column (Tag Editor) =====
        self._create_tag_editor()
        
        # ===== Bottom Bar (Progress) =====
        self._create_bottom_bar()
    
    def _create_top_bar(self):
        """Create folder selection bar."""
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 5))
        
        label = ctk.CTkLabel(top_frame, text="Target Folder:", font=("", 13, "bold"))
        label.pack(side="left", padx=(10, 5))
        
        self.folder_entry = ctk.CTkEntry(top_frame, placeholder_text="Select a folder...")
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        self.browse_btn = ctk.CTkButton(top_frame, text="Browse", width=80, command=self._browse_folder)
        self.browse_btn.pack(side="left", padx=5)
        
        self.refresh_btn = ctk.CTkButton(top_frame, text="↻", width=40, command=self._refresh_folder)
        self.refresh_btn.pack(side="left", padx=(0, 10))
    
    def _create_image_browser(self):
        """Create left column image browser."""
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        
        # Header
        header = ctk.CTkFrame(left_frame, fg_color="transparent")
        header.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(header, text="Images", font=("", 14, "bold")).pack(side="left")
        
        self.image_count_label = ctk.CTkLabel(header, text="(0)", font=("", 12), text_color="gray")
        self.image_count_label.pack(side="left", padx=5)
        
        # Selection buttons
        btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5)
        
        ctk.CTkButton(btn_frame, text="Select All", width=70, height=25, 
                      command=self._select_all).pack(side="left", padx=(0, 5))
        ctk.CTkButton(btn_frame, text="Clear", width=50, height=25,
                      command=self._clear_selection).pack(side="left")
        
        # Scrollable thumbnail list
        self.thumb_scroll = ctk.CTkScrollableFrame(left_frame)
        self.thumb_scroll.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_settings_panel(self):
        """Create center column settings panel."""
        center_frame = ctk.CTkScrollableFrame(self)
        center_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # ===== Model Selection =====
        self._create_model_section(center_frame)
        
        # ===== Local Model Config =====
        self._create_local_config_section(center_frame)
        
        # ===== API Configs =====
        self._create_api_config_section(center_frame)
        self._create_xai_config_section(center_frame)
        self._create_openrouter_config_section(center_frame)
        
        # ===== Model Settings =====
        self._create_settings_section(center_frame)
        
        # ===== Prompt Template =====
        self._create_template_section(center_frame)
        
        # ===== System Prompt =====
        self._create_prompt_section(center_frame)
        
        # ===== Controls =====
        self._create_controls_section(center_frame)
        
        # Initial visibility
        self._update_config_visibility()
    
    def _create_model_section(self, parent):
        """Create model type selection."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(section, text="Model Type", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        row = ctk.CTkFrame(section, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 10))
        
        self.model_type_var = ctk.StringVar(value="gemini")
        
        self.model_type_combo = ctk.CTkComboBox(
            row,
            values=list(MODEL_TYPE_OPTIONS.keys()),
            width=200,
            state="readonly",
            command=self._on_model_type_change,
        )
        self.model_type_combo.set("Gemini API")
        self.model_type_combo.pack(side="left")
    
    def _on_model_type_change(self, display_name: str):
        """Handle model type dropdown change."""
        self.model_type_var.set(MODEL_TYPE_OPTIONS[display_name])
        self._update_config_visibility()
    
    def _create_local_config_section(self, parent):
        """Create local VLM configuration."""
        self.local_section = ctk.CTkFrame(parent)
        self.local_section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.local_section, text="Local Model", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # VLM Type selector
        row0 = ctk.CTkFrame(self.local_section, fg_color="transparent")
        row0.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(row0, text="VLM Type:", width=80, anchor="w").pack(side="left")
        self.vlm_type_combo = ctk.CTkComboBox(row0, values=["Qwen3VL", "LLaVA"], width=120)
        self.vlm_type_combo.set("Qwen3VL")
        self.vlm_type_combo.pack(side="left")
        
        # Model file
        row1 = ctk.CTkFrame(self.local_section, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(row1, text="Model:", width=80, anchor="w").pack(side="left")
        self.model_path_entry = ctk.CTkEntry(row1, placeholder_text="Select .gguf file...")
        self.model_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(row1, text="...", width=30, command=self._browse_model).pack(side="right")
        
        # Mmproj file
        row2 = ctk.CTkFrame(self.local_section, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(row2, text="Projector:", width=80, anchor="w").pack(side="left")
        self.mmproj_path_entry = ctk.CTkEntry(row2, placeholder_text="Select mmproj.gguf...")
        self.mmproj_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(row2, text="...", width=30, command=self._browse_mmproj).pack(side="right")
        
        # Load button
        row3 = ctk.CTkFrame(self.local_section, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=(5, 10))
        
        self.load_model_btn = ctk.CTkButton(row3, text="Load Model", command=self._load_local_model)
        self.load_model_btn.pack(side="left")
        
        self.model_status_label = ctk.CTkLabel(row3, text="No model loaded", text_color="gray")
        self.model_status_label.pack(side="left", padx=(10, 0))
    
    def _create_api_config_section(self, parent):
        """Create Gemini API configuration."""
        self.api_section = ctk.CTkFrame(parent)
        self.api_section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.api_section, text="Gemini API", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # API Key
        row1 = ctk.CTkFrame(self.api_section, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(row1, text="API Key:", width=80, anchor="w").pack(side="left")
        self.api_key_entry = ctk.CTkEntry(row1, placeholder_text="Enter API key...", show="•")
        self.api_key_entry.pack(side="left", fill="x", expand=True)
        
        # Model selection
        row2 = ctk.CTkFrame(self.api_section, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(row2, text="Model:", width=80, anchor="w").pack(side="left")
        self.gemini_model_combo = ctk.CTkComboBox(row2, values=GEMINI_MODELS, width=180)
        self.gemini_model_combo.set(GEMINI_MODELS[0])
        self.gemini_model_combo.pack(side="left")
    
    def _create_xai_config_section(self, parent):
        """Create xAI Grok API configuration."""
        self.xai_section = ctk.CTkFrame(parent)
        self.xai_section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.xai_section, text="xAI (Grok)", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # API Key
        row1 = ctk.CTkFrame(self.xai_section, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(row1, text="API Key:", width=80, anchor="w").pack(side="left")
        self.xai_key_entry = ctk.CTkEntry(row1, placeholder_text="Enter xAI API key...", show="•")
        self.xai_key_entry.pack(side="left", fill="x", expand=True)
        
        # Model selection (editable combobox for custom input)
        row2 = ctk.CTkFrame(self.xai_section, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(row2, text="Model:", width=80, anchor="w").pack(side="left")
        self.xai_model_combo = ctk.CTkComboBox(row2, values=XAI_MODELS, width=220)
        self.xai_model_combo.set(XAI_MODELS[0])
        self.xai_model_combo.pack(side="left")
    
    def _create_openrouter_config_section(self, parent):
        """Create OpenRouter API configuration."""
        self.openrouter_section = ctk.CTkFrame(parent)
        self.openrouter_section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.openrouter_section, text="OpenRouter", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # API Key
        row1 = ctk.CTkFrame(self.openrouter_section, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(row1, text="API Key:", width=80, anchor="w").pack(side="left")
        self.openrouter_key_entry = ctk.CTkEntry(row1, placeholder_text="Enter OpenRouter API key...", show="•")
        self.openrouter_key_entry.pack(side="left", fill="x", expand=True)
        
        # Model selection (editable combobox for custom input)
        row2 = ctk.CTkFrame(self.openrouter_section, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(row2, text="Model:", width=80, anchor="w").pack(side="left")
        self.openrouter_model_combo = ctk.CTkComboBox(row2, values=OPENROUTER_MODELS, width=320)
        self.openrouter_model_combo.set(OPENROUTER_MODELS[0])
        self.openrouter_model_combo.pack(side="left")
    

    
    def _create_settings_section(self, parent):
        """Create generation settings with sliders."""
        self.settings_section = ctk.CTkFrame(parent)
        self.settings_section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.settings_section, text="Generation Settings", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        settings_frame = ctk.CTkFrame(self.settings_section, fg_color="transparent")
        settings_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        # Common settings
        self.temp_slider, self.temp_label = self._create_slider(settings_frame, "Temperature:", 0, 2, 0.4, 0)
        self.topk_slider, self.topk_label = self._create_slider(settings_frame, "Top-K:", 1, 100, 40, 1, True)
        self.topp_slider, self.topp_label = self._create_slider(settings_frame, "Top-P:", 0, 1, 0.9, 2)
        
        # Local VLM only settings
        self.local_settings_frame = ctk.CTkFrame(self.settings_section, fg_color="transparent")
        self.local_settings_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.local_settings_frame, text="Local VLM Only:", font=("", 11), text_color="gray").grid(row=0, column=0, columnspan=3, sticky="w")
        
        self.minp_slider, self.minp_label = self._create_slider(self.local_settings_frame, "Min-P:", 0, 1, 0.05, 1)
        self.repeat_slider, self.repeat_label = self._create_slider(self.local_settings_frame, "Repeat Penalty:", 0, 2, 1.1, 2)
        
        # Reasoning effort (xAI / OpenRouter only)
        self.reasoning_frame = ctk.CTkFrame(self.settings_section, fg_color="transparent")
        self.reasoning_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.reasoning_frame, text="API Reasoning (Gemini / xAI / OpenRouter):", font=("", 11), text_color="gray").pack(anchor="w")
        
        reason_row = ctk.CTkFrame(self.reasoning_frame, fg_color="transparent")
        reason_row.pack(fill="x", pady=3)
        
        ctk.CTkLabel(reason_row, text="Reasoning:", width=110, anchor="w").pack(side="left")
        self.reasoning_combo = ctk.CTkComboBox(
            reason_row,
            values=["none", "minimal", "low", "medium", "high", "auto"],
            width=120,
            state="readonly",
        )
        self.reasoning_combo.set("none")
        self.reasoning_combo.pack(side="left", padx=5)
    
    def _create_slider(self, parent, label_text: str, from_: float, to: float, default: float, row: int, is_int: bool = False):
        """Create a labeled slider."""
        label = ctk.CTkLabel(parent, text=label_text, width=110, anchor="w")
        label.grid(row=row, column=0, sticky="w", pady=3)
        
        value_label = ctk.CTkLabel(parent, text=f"{default:.2f}" if not is_int else str(int(default)), width=45)
        value_label.grid(row=row, column=2, padx=(5, 0))
        
        def on_change(value):
            if is_int:
                value_label.configure(text=str(int(float(value))))
            else:
                value_label.configure(text=f"{float(value):.2f}")
        
        slider = ctk.CTkSlider(parent, from_=from_, to=to, command=on_change, width=200)
        slider.set(default)
        slider.grid(row=row, column=1, padx=5, pady=3)
        
        return slider, value_label
    
    def _create_template_section(self, parent):
        """Create prompt template selector."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(section, text="Prompt Template", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        row = ctk.CTkFrame(section, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 10))
        
        self.template_combo = ctk.CTkComboBox(
            row, values=self.templates.get_names(), 
            width=200, command=self._on_template_change
        )
        self.template_combo.set(self.templates.get_names()[0] if self.templates.get_names() else "")
        self.template_combo.pack(side="left", padx=(0, 10))
        
        ctk.CTkButton(row, text="Save", width=50, command=self._quick_save_template).pack(side="left", padx=(0, 5))
        ctk.CTkButton(row, text="Save As...", width=70, command=self._save_template).pack(side="left", padx=(0, 5))
        ctk.CTkButton(row, text="Delete", width=60, fg_color="red", hover_color="darkred",
                      command=self._delete_template).pack(side="left")
    
    def _create_prompt_section(self, parent):
        """Create system prompt text area."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="both", expand=True, pady=(0, 10))
        
        ctk.CTkLabel(section, text="System Prompt", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.prompt_text = ctk.CTkTextbox(section, height=200)
        self.prompt_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.prompt_text.insert("1.0", self.config.get("system_prompt"))
    
    def _create_controls_section(self, parent):
        """Create start/stop controls."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=(0, 10))
        
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶ Start Tagging", 
            fg_color="green", hover_color="darkgreen",
            font=("", 14, "bold"),
            height=40,
            command=self._start_processing
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.stop_btn = ctk.CTkButton(
            btn_frame, text="■ Stop",
            fg_color="red", hover_color="darkred",
            height=40,
            width=80,
            command=self._stop_processing,
            state="disabled"
        )
        self.stop_btn.pack(side="left")
    
    def _create_tag_editor(self):
        """Create right column tag editor."""
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=5)
        
        # Header
        header = ctk.CTkFrame(right_frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(header, text="Tag Editor", font=("", 14, "bold")).pack(side="left")
        
        self.editor_filename_label = ctk.CTkLabel(header, text="", font=("", 12), text_color="gray")
        self.editor_filename_label.pack(side="left", padx=10)
        
        # Image preview
        self.preview_frame = ctk.CTkFrame(right_frame, height=200)
        self.preview_frame.pack(fill="x", padx=10, pady=5)
        self.preview_frame.pack_propagate(False)
        
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Select an image to preview")
        self.preview_label.pack(expand=True)
        
        # Text editor
        self.tag_editor = ctk.CTkTextbox(right_frame, height=200)
        self.tag_editor.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Editor buttons
        btn_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.save_tags_btn = ctk.CTkButton(
            btn_frame, text="💾 Save", width=80,
            command=self._save_current_tags
        )
        self.save_tags_btn.pack(side="left", padx=(0, 5))
        
        self.revert_tags_btn = ctk.CTkButton(
            btn_frame, text="↩ Revert", width=80,
            fg_color="gray", hover_color="darkgray",
            command=self._revert_current_tags
        )
        self.revert_tags_btn.pack(side="left")
    
    def _create_bottom_bar(self):
        """Create progress bar at bottom."""
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(5, 10))
        
        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=(10, 5))
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(bottom_frame, text="Ready")
        self.status_label.pack(anchor="w", padx=10, pady=(0, 10))
    
    # ===== Event Handlers =====
    
    def _setup_callbacks(self):
        """Setup tagger callbacks."""
        tagger = get_tagger()
        tagger.on_progress = self._on_progress
        tagger.on_error = self._on_error
        tagger.on_complete = self._on_complete
    
    def _update_config_visibility(self):
        """Show/hide config sections based on model type."""
        mode = self.model_type_var.get()
        
        # Hide all config sections first
        self.local_section.pack_forget()
        self.api_section.pack_forget()
        self.xai_section.pack_forget()
        self.openrouter_section.pack_forget()
        self.local_settings_frame.pack_forget()
        
        # Show the relevant section
        if mode == "local":
            self.local_section.pack(fill="x", pady=(0, 10), before=self.settings_section)
            self.local_settings_frame.pack(fill="x", padx=10, pady=(0, 10))
            self.reasoning_frame.pack_forget()
        elif mode == "gemini":
            self.api_section.pack(fill="x", pady=(0, 10), before=self.settings_section)
            self.reasoning_frame.pack(fill="x", padx=10, pady=(0, 10))
        elif mode == "xai":
            self.xai_section.pack(fill="x", pady=(0, 10), before=self.settings_section)
            self.reasoning_frame.pack(fill="x", padx=10, pady=(0, 10))
        elif mode == "openrouter":
            self.openrouter_section.pack(fill="x", pady=(0, 10), before=self.settings_section)
            self.reasoning_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    def _browse_folder(self):
        """Open folder browser."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, folder)
            self._load_folder(folder)
    
    def _refresh_folder(self):
        """Refresh current folder."""
        folder = self.folder_entry.get().strip()
        if folder:
            self._load_folder(folder)
    
    def _load_folder(self, folder: str):
        """Load images from folder."""
        # Clear existing thumbnails
        for thumb in self._thumbnails:
            thumb.destroy()
        self._thumbnails.clear()
        self._selected_thumbnail = None
        
        # Find images
        self._current_images = find_images(folder)
        self.image_count_label.configure(text=f"({len(self._current_images)})")
        
        # Create thumbnails
        for img_path in self._current_images:
            thumb = ImageThumbnail(
                self.thumb_scroll, 
                img_path, 
                on_select=self._on_thumbnail_select
            )
            thumb.pack(fill="x", pady=1)
            self._thumbnails.append(thumb)
        
        # Select first if available
        if self._thumbnails:
            self._on_thumbnail_select(self._thumbnails[0])
    
    def _on_thumbnail_select(self, thumbnail: ImageThumbnail):
        """Handle thumbnail selection."""
        # Deselect previous
        if self._selected_thumbnail:
            self._selected_thumbnail.set_selected(False)
        
        # Select new
        self._selected_thumbnail = thumbnail
        thumbnail.set_selected(True)
        
        # Update editor
        self._load_image_to_editor(thumbnail.image_path)
    
    def _load_image_to_editor(self, image_path: Path):
        """Load image and its tags to editor."""
        # Update filename label
        self.editor_filename_label.configure(text=image_path.name)
        
        # Load preview image
        try:
            img = Image.open(image_path)
            # Scale to fit preview area
            max_size = (330, 180)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            self.preview_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as e:
            self.preview_label.configure(text=f"Error: {e}", image=None)
        
        # Load tags
        txt_path = get_output_path(image_path)
        self.tag_editor.delete("1.0", "end")
        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    self.tag_editor.insert("1.0", f.read())
            except Exception as e:
                self.tag_editor.insert("1.0", f"Error loading: {e}")
    
    def _save_current_tags(self):
        """Save current editor content to file."""
        if not self._selected_thumbnail:
            return
        
        txt_path = get_output_path(self._selected_thumbnail.image_path)
        content = self.tag_editor.get("1.0", "end-1c")
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self._selected_thumbnail.update_status()
            self.status_label.configure(text=f"Saved: {txt_path.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def _revert_current_tags(self):
        """Revert editor content from file."""
        if self._selected_thumbnail:
            self._load_image_to_editor(self._selected_thumbnail.image_path)
    
    def _select_all(self):
        """Select all images for tagging."""
        for thumb in self._thumbnails:
            thumb.checkbox_var.set(True)
    
    def _clear_selection(self):
        """Clear all selections."""
        for thumb in self._thumbnails:
            thumb.checkbox_var.set(False)
    
    def _browse_model(self):
        """Browse for model file."""
        file = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if file:
            self.model_path_entry.delete(0, "end")
            self.model_path_entry.insert(0, file)
    
    def _browse_mmproj(self):
        """Browse for mmproj file."""
        file = filedialog.askopenfilename(
            title="Select Projector File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if file:
            self.mmproj_path_entry.delete(0, "end")
            self.mmproj_path_entry.insert(0, file)
    
    def _load_local_model(self):
        """Load local VLM model."""
        model_path = self.model_path_entry.get().strip()
        mmproj_path = self.mmproj_path_entry.get().strip()
        
        if not model_path or not mmproj_path:
            messagebox.showerror("Error", "Please select both model and projector files.")
            return
        
        # Get VLM type from selector
        vlm_type_str = self.vlm_type_combo.get()
        vlm_type = VLMType.QWEN3VL if vlm_type_str == "Qwen3VL" else VLMType.LLAVA
        
        self.model_status_label.configure(text="Loading...", text_color="yellow")
        self.update()
        
        def load():
            vlm = get_local_vlm()
            success = vlm.load_model(model_path, mmproj_path, model_type=vlm_type)
            self.after(0, lambda: self._on_model_loaded(success))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _on_model_loaded(self, success: bool):
        """Callback when model loading completes."""
        if success:
            self.model_status_label.configure(text="Model loaded ✓", text_color="green")
        else:
            self.model_status_label.configure(text="Failed to load", text_color="red")
    
    def _on_template_change(self, name: str):
        """Handle template selection change."""
        prompt = self.templates.get_prompt(name)
        if prompt:
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", prompt)
    
    def _quick_save_template(self):
        """Quick save: overwrite current template with current prompt text."""
        name = self.template_combo.get()
        if not name:
            self._save_template()
            return
        
        if self.templates.is_default(name):
            messagebox.showinfo("Info", f"'{name}' is a default template.\nUse 'Save As...' to create a copy.")
            return
        
        prompt = self.prompt_text.get("1.0", "end-1c")
        if self.templates.add(name, prompt):
            self.status_label.configure(text=f"Template saved: {name}")
    
    def _save_template(self):
        """Save current prompt as new template."""
        dialog = ctk.CTkInputDialog(text="Enter template name:", title="Save Template")
        name = dialog.get_input()
        
        if name:
            prompt = self.prompt_text.get("1.0", "end-1c")
            if self.templates.add(name, prompt):
                self.template_combo.configure(values=self.templates.get_names())
                self.template_combo.set(name)
                self.status_label.configure(text=f"Template saved: {name}")
    
    def _delete_template(self):
        """Delete current template."""
        name = self.template_combo.get()
        if self.templates.is_default(name):
            messagebox.showwarning("Warning", "Cannot delete default templates.")
            return
        
        if messagebox.askyesno("Confirm", f"Delete template '{name}'?"):
            if self.templates.delete(name):
                self.template_combo.configure(values=self.templates.get_names())
                if self.templates.get_names():
                    self.template_combo.set(self.templates.get_names()[0])
    
    def _get_settings(self) -> dict:
        """Get current settings from UI."""
        return {
            "temperature": self.temp_slider.get(),
            "top_k": int(self.topk_slider.get()),
            "top_p": self.topp_slider.get(),
            "min_p": self.minp_slider.get(),
            "repeat_penalty": self.repeat_slider.get(),
        }
    
    def _start_processing(self):
        """Start image processing."""
        folder = self.folder_entry.get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        
        # Get checked images
        checked_images = [t.image_path for t in self._thumbnails if t.is_checked()]
        if not checked_images:
            messagebox.showerror("Error", "No images selected for tagging.")
            return
        
        # Configure tagger
        tagger = get_tagger()
        
        # Set backend
        mode = self.model_type_var.get()
        
        if mode == "local":
            tagger.backend_type = BackendType.LOCAL_VLM
            if not get_local_vlm().is_loaded():
                messagebox.showerror("Error", "Please load a local model first.")
                return
        elif mode == "gemini":
            tagger.backend_type = BackendType.GEMINI_API
            api_key = self.api_key_entry.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your Gemini API key.")
                return
            
            gemini = get_gemini_api()
            if not gemini.configure(api_key, self.gemini_model_combo.get()):
                messagebox.showerror("Error", "Failed to configure Gemini API.")
                return
        elif mode == "xai":
            tagger.backend_type = BackendType.XAI
            api_key = self.xai_key_entry.get().strip()
            model_name = self.xai_model_combo.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your xAI API key.")
                return
            if not model_name:
                messagebox.showerror("Error", "Please select or enter an xAI model.")
                return
            
            xai = get_xai_api()
            if not xai.configure(api_key, model_name, XAI_BASE_URL):
                messagebox.showerror("Error", "Failed to configure xAI API.")
                return
        elif mode == "openrouter":
            tagger.backend_type = BackendType.OPENROUTER
            api_key = self.openrouter_key_entry.get().strip()
            model_name = self.openrouter_model_combo.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your OpenRouter API key.")
                return
            if not model_name:
                messagebox.showerror("Error", "Please select or enter an OpenRouter model.")
                return
            
            openrouter = get_openrouter_api()
            if not openrouter.configure(api_key, model_name, OPENROUTER_BASE_URL):
                messagebox.showerror("Error", "Failed to configure OpenRouter API.")
                return
        
        # Set format and prompt from template
        template_name = self.template_combo.get()
        template_format = self.templates.get_format(template_name)
        tagger.format = TaggingFormat.CAPTIONING if template_format == "captioning" else TaggingFormat.TAG
        tagger.system_prompt = self.prompt_text.get("1.0", "end-1c").strip()
        
        # Set generation settings
        settings = self._get_settings()
        tagger.temperature = settings["temperature"]
        tagger.top_k = settings["top_k"]
        tagger.top_p = settings["top_p"]
        tagger.min_p = settings["min_p"]
        tagger.repeat_penalty = settings["repeat_penalty"]
        tagger.reasoning_effort = self.reasoning_combo.get()
        
        # Store checked images for processing
        self._images_to_process = checked_images
        
        # Update UI
        self._processing = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)
        self.status_label.configure(text="Starting...")
        
        # Start processing thread
        def process():
            tagger.process_images(self._images_to_process, self._on_image_tagged)
        
        self._process_thread = threading.Thread(target=process, daemon=True)
        self._process_thread.start()
    
    def _on_image_tagged(self, image_path: Path):
        """Callback when a single image is tagged."""
        def update():
            # Update thumbnail status
            for thumb in self._thumbnails:
                if thumb.image_path == image_path:
                    thumb.update_status()
                    # If this is the selected image, refresh editor
                    if thumb == self._selected_thumbnail:
                        self._load_image_to_editor(image_path)
                    break
        
        self.after(0, update)
    
    def _stop_processing(self):
        """Stop image processing."""
        get_tagger().stop()
        self.status_label.configure(text="Stopping...")
    
    def _on_progress(self, current: int, total: int, filename: str):
        """Progress callback."""
        def update():
            self.progress_bar.set(current / total)
            self.status_label.configure(text=f"Processing ({current}/{total}): {filename}")
        self.after(0, update)
    
    def _on_error(self, filename: str, error: str):
        """Error callback."""
        def update():
            if filename:
                self.status_label.configure(text=f"Error on {filename}: {error}")
            else:
                self.status_label.configure(text=f"Error: {error}")
        self.after(0, update)
    
    def _on_complete(self, processed: int):
        """Completion callback."""
        def update():
            self._processing = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.progress_bar.set(1 if processed > 0 else 0)
            self.status_label.configure(text=f"Complete! Processed {processed} images.")
        self.after(0, update)
    
    # ===== Settings Persistence =====
    
    def _load_settings(self):
        """Load settings from config."""
        config = self.config
        
        # Folder
        last_folder = config.get("last_folder")
        if last_folder:
            self.folder_entry.insert(0, last_folder)
            if Path(last_folder).is_dir():
                self._load_folder(last_folder)
        
        # Model type
        model_type_val = config.get("model_type", "gemini")
        self.model_type_var.set(model_type_val)
        display_name = MODEL_TYPE_DISPLAY.get(model_type_val, "Gemini API")
        self.model_type_combo.set(display_name)
        self._update_config_visibility()
        
        # Gemini API settings
        api_key = config.get("api_key")
        if api_key:
            self.api_key_entry.insert(0, api_key)
        self.gemini_model_combo.set(config.get("gemini_model", GEMINI_MODELS[0]))
        
        # xAI settings
        xai_key = config.get("xai_api_key")
        if xai_key:
            self.xai_key_entry.insert(0, xai_key)
        xai_model = config.get("xai_model")
        if xai_model:
            self.xai_model_combo.set(xai_model)
        
        # OpenRouter settings
        or_key = config.get("openrouter_api_key")
        if or_key:
            self.openrouter_key_entry.insert(0, or_key)
        or_model = config.get("openrouter_model")
        if or_model:
            self.openrouter_model_combo.set(or_model)
        
        # Local model paths
        local_model = config.get("local_model_path")
        if local_model:
            self.model_path_entry.insert(0, local_model)
        local_mmproj = config.get("local_mmproj_path")
        if local_mmproj:
            self.mmproj_path_entry.insert(0, local_mmproj)
        
        # VLM Type
        vlm_type = config.get("vlm_type", "Qwen3VL")
        self.vlm_type_combo.set(vlm_type)
        
        # Sliders — set values and update labels
        temp_val = config.get("temperature", 0.4)
        self.temp_slider.set(temp_val)
        self.temp_label.configure(text=f"{temp_val:.2f}")
        
        topk_val = config.get("top_k", 40)
        self.topk_slider.set(topk_val)
        self.topk_label.configure(text=str(int(topk_val)))
        
        topp_val = config.get("top_p", 0.9)
        self.topp_slider.set(topp_val)
        self.topp_label.configure(text=f"{topp_val:.2f}")
        
        minp_val = config.get("min_p", 0.05)
        self.minp_slider.set(minp_val)
        self.minp_label.configure(text=f"{minp_val:.2f}")
        
        repeat_val = config.get("repeat_penalty", 1.1)
        self.repeat_slider.set(repeat_val)
        self.repeat_label.configure(text=f"{repeat_val:.2f}")
        
        # Reasoning effort
        reasoning = config.get("reasoning_effort", "none")
        self.reasoning_combo.set(reasoning)
        
        # Template
        template_name = config.get("selected_template")
        if template_name and template_name in self.templates.get_names():
            self.template_combo.set(template_name)
            self._on_template_change(template_name)
    
    def _save_settings(self):
        """Save current settings to config."""
        self.config.update({
            "last_folder": self.folder_entry.get().strip(),
            "model_type": self.model_type_var.get(),
            "gemini_model": self.gemini_model_combo.get(),
            "api_key": self.api_key_entry.get().strip(),
            "xai_api_key": self.xai_key_entry.get().strip(),
            "xai_model": self.xai_model_combo.get(),
            "openrouter_api_key": self.openrouter_key_entry.get().strip(),
            "openrouter_model": self.openrouter_model_combo.get(),
            "local_model_path": self.model_path_entry.get().strip(),
            "local_mmproj_path": self.mmproj_path_entry.get().strip(),
            "vlm_type": self.vlm_type_combo.get(),
            "temperature": self.temp_slider.get(),
            "top_k": int(self.topk_slider.get()),
            "top_p": self.topp_slider.get(),
            "min_p": self.minp_slider.get(),
            "repeat_penalty": self.repeat_slider.get(),
            "reasoning_effort": self.reasoning_combo.get(),
            "selected_template": self.template_combo.get(),
            "system_prompt": self.prompt_text.get("1.0", "end-1c").strip(),
            "window_geometry": f"{self.winfo_width()}x{self.winfo_height()}",
        })
        self.config.save()
    
    def _on_close(self):
        """Handle window close."""
        self._save_settings()
        self.destroy()


def run_app():
    """Run the application."""
    app = ImageTaggerApp()
    app.mainloop()
