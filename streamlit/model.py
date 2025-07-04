"""
Model factory with:

• ResNet-variant autoselection (18/34/50/101) from checkpoint keys
• Local-file first, optional fallback download via AppConfig.resolved_model_path
• Automatic CUDA half-precision (cfg.half_precision)
• Synthetic warm-up + FPS benchmark (cfg.batch_size)
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision import models

from config import cfg, AppConfig

# ------------------------------------------------------------------- #
# I.  model introspection utils
# ------------------------------------------------------------------- #
def _strip_module(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state.items()}


def _infer_resnet_variant(state: Dict[str, torch.Tensor]) -> str:
    bottleneck = any(".conv3.weight" in k for k in state)
    l3 = [int(k.split(".")[1]) for k in state if k.startswith("layer3.") and ".conv1" in k]
    max_blk = max(l3) if l3 else 1
    if not bottleneck:
        return "resnet34" if max_blk >= 5 else "resnet18"
    return "resnet101" if max_blk >= 22 else "resnet50"


def _build_resnet(variant: str, num_classes: int) -> nn.Module:
    factory = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }[variant]
    model = factory(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------------------------------------------------- #
# II.  public loader
# ------------------------------------------------------------------- #
def load_emotion_model(settings: AppConfig = cfg) -> Tuple[nn.Module, torch.device]:
    ckpt_path = settings.resolved_model_path
    device = torch.device("cuda" if settings.gpu and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    ckpt = _strip_module(ckpt)

    var = _infer_resnet_variant(ckpt)
    print(f"[EmotionSense] detected {var} checkpoint")

    model = _build_resnet(var, len(settings.emotion_labels))
    own = model.state_dict()
    matched = {k: v for k, v in ckpt.items() if k in own and v.shape == own[k].shape}
    own.update(matched)
    model.load_state_dict(own, strict=False)
    print(f"[EmotionSense] loaded {len(matched)}/{len(own)} tensors")

    model.to(device)
    if settings.half_precision and device.type == "cuda":
        model.half()
        amp_ctx = torch.cuda.amp.autocast
    else:
        amp_ctx = nullcontext

    # ---------- warm-up & FPS ----------
    bs = settings.batch_size
    dummy = torch.randn(bs, 3, settings.input_size, settings.input_size, device=device)
    if settings.half_precision and device.type == "cuda":
        dummy = dummy.half()

    with torch.no_grad(), amp_ctx():
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fps = bs / (perf_counter() - t0)
    print(f"[EmotionSense] warm-up complete ⇒ {fps:.1f} inferences/sec (batch={bs})")

    model.eval()
    return model, device


# ------------------------------------------------------------------- #
# III.  torchvision transform
# ------------------------------------------------------------------- #
from torchvision import transforms as T

def build_transform(settings=cfg) -> T.Compose:
    s = settings.input_size
    return T.Compose([
        T.Resize(int(s * 1.14)),
        T.CenterCrop(s),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
