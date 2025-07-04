# streamlit/config.py

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
    "accent": "#FF5A00",
}

_PROFESSIONAL = {
    "background": "#FDFDFD",
    "card": "#FFFFFF",
    "accent": "#007BFF",
    "text": "#2C3E50",
    "secondary": "#ECF0F1",
    "success": "#28A745",
    "warning": "#FFC107",
    "error": "#DC3545",
    "border": "#E0E0E0",
    "chart_colors": [
        "#007BFF",
        "#28A745",
        "#FFC107",
        "#DC3545",
        "#6F42C1",
        "#17A2B8",
        "#6C757D",
    ],
}

# ───────────────────────── settings ──────────────────────────
class AppConfig(BaseSettings):
    """
    Centralised, declarative configuration with env-override support.
    """

    # ----- UI -----
    theme: Literal["light", "dark", "corp", "pro"] = Field("pro", env="THEME")
    confidence: float = Field(0.7, ge=0, le=1, env="CONFIDENCE")
    input_size: int = Field(224, gt=0, env="INPUT_SIZE")

    # ----- model -----
    model_path: Path = Field(
        default=Path(
            r"..\checkpoints\emotion_model.pth"
        ),
        env="MODEL_PATH",
    )
    model_url: str | None = Field(
        default="https://drive.google.com/uc?export=download&id=1yGdQQsoskjAOG-IG9OoFS3K2aBWyVDcD",
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
        env_prefix = "EMOTIONSENSE_"
        env_file = ".env"
        frozen = True
        case_sensitive = False

    # ---------- derived props ----------
    @property
    def palette(self) -> dict[str, str | list[str]]:
        return {
            "light": _LIGHT,
            "dark": _DARK,
            "corp": _CORP,
            "pro": _PROFESSIONAL,
        }[self.theme]

    # ---------- helpers ----------
    def to_json(self, path: Path | None = None) -> str:
        data = self.model_dump()
        data["palette"] = self.palette
        text = json.dumps(data, indent=2)
        if path:
            path.write_text(text)
        return text

    @property
    def resolved_model_path(self) -> Path:
        """
        Return a valid checkpoint path, downloading from model_url if not found locally.
        """
        if self.model_path.exists():
            return self.model_path

        # fallback: download from Google Drive (converted link)
        if self.model_url:
            import urllib.request

            checkpoints_dir = self.model_path.parent
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            tmp_path = checkpoints_dir / "downloaded_model.pth"
            print(f"[INFO] Checkpoint not found locally, downloading from {self.model_url}...")

            try:
                urllib.request.urlretrieve(self.model_url, tmp_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download model from {self.model_url}: {e}")

            return tmp_path

        raise FileNotFoundError(
            f"Model checkpoint not found at {self.model_path} and no MODEL_URL provided."
        )

    @validator("model_path")
    def _expand_path(cls, p: Path) -> Path:
        return p.expanduser().resolve()


cfg = AppConfig()
