# C:\Users\alvar\Documents\GitHub\emotion_recognition\streamlit\config.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

try:
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:
    from pydantic import BaseSettings
from pydantic import Field, validator
import torch

# ───────────────────────── palettes ──────────────────────────
# Existing palettes (keeping them for reference, though _PROFESSIONAL will be default)
_LIGHT = {
    "background": "#FFFFFF",
    "card": "#F5F7FA",
    "accent": "#4A76D0",
    "text": "#2D3748",
    "secondary": "#E2E8F0",
    "success": "#48BB78",
    "warning": "#ED8936",
    "error": "#E53E3E",
    "border": "#CBD5E0",
}

_DARK = {
    "background": "#1A202C",
    "card": "#2D3748",
    "accent": "#63B3ED",
    "text": "#E2E8F0",
    "secondary": "#4A5568",
    "success": "#48BB78",
    "warning": "#ED8936",
    "error": "#FC8181",
    "border": "#4A5568",
}

_CORP = {
    **_LIGHT,
    "accent": "#FF5A00",  # example corporate orange
}

# --- IMPROVED PROFESSIONAL PALETTE ---
_PROFESSIONAL = {
    "background": "#FDFDFD",      # Very light off-white for main background
    "card": "#FFFFFF",            # Pure white for cards, creating a clean, crisp look
    "accent": "#007BFF",          # A vibrant, professional blue for key elements
    "text": "#2C3E50",            # Dark, soft gray for primary text, easy on the eyes
    "secondary": "#ECF0F1",       # Light gray for secondary backgrounds/dividers/highlights
    "success": "#28A745",         # Standard green for success, clear and direct
    "warning": "#FFC107",         # Standard amber for warning, noticeable
    "error": "#DC3545",           # Standard red for error, for critical alerts
    "border": "#E0E0E0",          # Soft border color for subtle separation
    # New: Chart specific colors for Pyecharts to ensure distinct and appealing visuals
    "chart_colors": [
        "#007BFF", # Accent blue
        "#28A745", # Success green
        "#FFC107", # Warning amber
        "#DC3545", # Error red
        "#6F42C1", # A professional purple
        "#17A2B8", # A calming teal
        "#6C757D"  # A neutral grey
    ]
}


# ───────────────────────── settings ──────────────────────────
class AppConfig(BaseSettings):
    """
    Centralised, declarative configuration with env-override support.

    • Reads values from environment variables (prefix `EMOTIONSENSE_`)
      or a local `.env` file – perfect for CI/CD or Streamlit Cloud.
    • Immutable (`frozen=True`) so downstream code can trust it.
    """

    # ----- UI -----
    theme: Literal["light", "dark", "corp", "pro"] = Field("pro", env="THEME") # Default to 'pro'
    confidence: float = Field(0.7, ge=0, le=1, env="CONFIDENCE")
    input_size: int = Field(224, gt=0, env="INPUT_SIZE")

    # ----- model / runtime -----
    model_path: Path = Field(
        default=Path(
            r"C:\Users\alvar\Documents\GitHub\emotion_recognition\checkpoints\emotion_model.pth"
        ),
        env="MODEL_PATH",
    )
    model_url: str | None = Field(
        default=None,  # optional remote fallback (HuggingFace, S3…)
        env="MODEL_URL",
    )
    batch_size: int = Field(8, gt=0, env="BATCH_SIZE")
    half_precision: bool = Field(False, env="HALF_PRECISION")
    gpu: bool = Field(default=torch.cuda.is_available(), env="USE_GPU")

    # ----- domain labels -----
    emotion_labels: tuple[str, ...] = (
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    )

    # pydantic config
    class Config:
        env_prefix = "EMOTIONSENSE_"  # e.g. EMOTIONSENSE_MODEL_PATH
        env_file = ".env"
        frozen = True
        case_sensitive = False

    # ---------- derived props ----------
    @property
    def palette(self) -> dict[str, str | list[str]]: # Updated return type to include list[str] for chart_colors
        return {
            "light": _LIGHT,
            "dark": _DARK,
            "corp": _CORP,
            "pro": _PROFESSIONAL,
        }[self.theme]

    # ---------- helpers ----------
    def to_json(self, path: Path | None = None) -> str:
        """Serialise settings (resolved palette included) to JSON."""
        data = self.model_dump() # Use model_dump for Pydantic v2+
        data["palette"] = self.palette
        text = json.dumps(data, indent=2)
        if path:
            path.write_text(text)
        return text

    @validator("model_path")
    def _expand_path(cls, p: Path) -> Path:  # noqa: N805
        return p.expanduser().resolve()


# single-point instantiation
cfg = AppConfig()