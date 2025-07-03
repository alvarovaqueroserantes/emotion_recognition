from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

# try ultra-fast MediaPipe; fall back to cvlib
try:
    import mediapipe as mp

    _MP_DETECTOR = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # short-range
        min_detection_confidence=0.4,
    )
    _USE_MEDIAPIPE = True
except ModuleNotFoundError:  # noqa: B019
    import cvlib as cv  # type: ignore

    _USE_MEDIAPIPE = False

from config import cfg
from model import build_transform


@dataclass(slots=True)
class DetectedFace:
    box: Tuple[int, int, int, int]
    label: str
    confidence: float
    probabilities: Sequence[float]


# ──────────────────────────────────────────────────
class EmotionDetector:
    """
    Fast inference pipeline:

    1. Face boxes via MediaPipe (≈ 200 fps on CPU) or cvlib fallback.
    2. **Batch** all faces into one tensor.
    3. Optional half-precision with autocast.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.transform = build_transform(cfg)
        self.labels = cfg.emotion_labels
        self.conf_thresh = cfg.confidence
        self.autocast = (
            torch.cuda.amp.autocast if cfg.half_precision and device.type == "cuda" else nullcontext
        )

    # ---------- public ------------------------------------------------
    def detect(self, bgr_frame: np.ndarray) -> List[DetectedFace]:
        faces = self._faces_mp(bgr_frame) if _USE_MEDIAPIPE else self._faces_cv(bgr_frame)

        crops, meta = self._preprocess_faces(bgr_frame, faces)
        if not crops:
            return []

        with torch.no_grad(), self.autocast():
            logits = self.model(torch.stack(crops).to(self.device))
            probs = torch.softmax(logits, 1).cpu().numpy()
            preds = probs.argmax(1)

        out: List[DetectedFace] = []
        for (box, conf), idx, p in zip(meta, preds, probs):
            out.append(DetectedFace(box, self.labels[int(idx)], conf, p.tolist()))
        return out

    def draw(self, bgr_frame: np.ndarray, detections: List[DetectedFace]) -> np.ndarray:
        frame = bgr_frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
            cv2.putText(
                frame,
                f"{det.label} ({det.confidence:.0%})",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        return frame

    # ---------- private helpers --------------------------------------
    @staticmethod
    def _faces_mp(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Return [(box, score), …] where box = (x1,y1,x2,y2)."""
        h, w, _ = frame.shape
        results = _MP_DETECTOR.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = []
        if results.detections:
            for det in results.detections:
                score = det.score[0]
                if score < cfg.confidence:
                    continue
                bbox = det.location_data.relative_bounding_box
                x1 = int(max(0, bbox.xmin * w))
                y1 = int(max(0, bbox.ymin * h))
                x2 = int(min(w, (bbox.xmin + bbox.width) * w))
                y2 = int(min(h, (bbox.ymin + bbox.height) * h))
                faces.append(((x1, y1, x2, y2), score))
        return faces

    @staticmethod
    def _faces_cv(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        boxes, confs = cv.detect_face(frame)  # type: ignore[attr-defined]
        return list(zip(boxes, confs))

    def _preprocess_faces(
        self,
        frame: np.ndarray,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
    ) -> Tuple[List[torch.Tensor], List[Tuple[Tuple[int, int, int, int], float]]]:
        crops: List[torch.Tensor] = []
        meta: List[Tuple[Tuple[int, int, int, int], float]] = []
        for box, conf in detections:
            if conf < self.conf_thresh:
                continue
            x1, y1, x2, y2 = box
            pad = 8
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(roi_rgb))
            crops.append(tensor)
            meta.append((box, conf))
        return crops, meta


# small no-op context for CPU / fp32 path
from contextlib import nullcontext  # at bottom to avoid stdlib reorder
