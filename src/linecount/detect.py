"""Object detection: RF-DETR (accurate) or YOLOX on OpenVINO (fast), both Apache-2.0 and pretrained on COCO.

The detector returns ``supervision.Detections`` in full-frame pixel coordinates,
keeping only the requested COCO classes. Two families of models:

* ``nano`` / ``small`` / ``medium``: RF-DETR, a transformer detector run with
  PyTorch. The most accurate option, but about 0.5 s per frame on a laptop CPU.
* ``yolox-nano`` / ``yolox-tiny`` / ``yolox-s``: YOLOX, a small convolutional
  detector, run with Intel's OpenVINO runtime. Less accurate, several times
  faster on the same CPU (``yolox-tiny``: about 0.1 s per frame), which is what
  makes live use on a laptop or a small box possible. ``device="GPU"`` runs it
  on Intel integrated graphics.

Options shared by both:

* ``roi``: run the model on a crop around the counting line instead of the whole
  frame. Objects in the crop are seen at a larger scale and the model input is
  smaller, so it is both more accurate for small objects and faster.

Weights are downloaded on first use, never into this repository: RF-DETR into
``RF_HOME`` (or rfdetr's default cache), YOLOX into ``LINECOUNT_HOME``, else
``RF_HOME``, else ``~/.cache/linecount``.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

VEHICLES = ("car", "truck", "bus", "motorcycle")
PEOPLE = ("person",)

_RFDETR = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}
_YOLOX = {"yolox-nano": "yolox_nano", "yolox-tiny": "yolox_tiny", "yolox-s": "yolox_s"}
MODELS = tuple(_RFDETR) + tuple(_YOLOX)
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"

# The 80 COCO classes in the order YOLOX predicts them.
COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Roi:
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def parse(cls, text: str) -> Roi:
        x1, y1, x2, y2 = (int(round(float(v))) for v in text.split(","))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"roi must be x1,y1,x2,y2 with x2>x1 and y2>y1, got {text!r}")
        return cls(x1, y1, x2, y2)

    def clip(self, width: int, height: int) -> Roi:
        return Roi(max(0, self.x1), max(0, self.y1), min(width, self.x2), min(height, self.y2))


def weights_dir() -> Path:
    base = os.environ.get("LINECOUNT_HOME") or os.environ.get("RF_HOME") or Path.home() / ".cache" / "linecount"
    return Path(base)


class Detector:
    def __init__(
        self,
        model: str = "nano",
        threshold: float = 0.5,
        classes: tuple[str, ...] = VEHICLES,
        roi: Roi | None = None,
        agnostic_nms: float | None = None,
        device: str = "CPU",
    ):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.model_name = model
        self.threshold = threshold
        self.classes = tuple(classes)
        self.roi = roi
        self.agnostic_nms = agnostic_nms
        if model in _RFDETR:
            import rfdetr  # imported here so the rest of the package (and the tests) do not need torch

            self._backend = _RFDETRBackend(getattr(rfdetr, _RFDETR[model])())
        else:
            self._backend = _YOLOXBackend(_YOLOX[model], device.upper())

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        h, w = frame_bgr.shape[:2]
        x0 = y0 = 0
        image = frame_bgr
        if self.roi is not None:
            r = self.roi.clip(w, h)
            image = frame_bgr[r.y1 : r.y2, r.x1 : r.x2]
            x0, y0 = r.x1, r.y1
        det = self._backend(image, self.threshold)
        names = np.asarray(det.data.get("class_name", []), dtype=object)
        keep = np.isin(names, self.classes) if len(det) else np.zeros(0, bool)
        det = det[keep]
        if len(det) == 0:
            return sv.Detections.empty()
        det.xyxy = det.xyxy + np.array([x0, y0, x0, y0], dtype=det.xyxy.dtype)
        if self.agnostic_nms is not None:
            det = det.with_nms(threshold=self.agnostic_nms, class_agnostic=True)
        return det


class _RFDETRBackend:
    def __init__(self, model):
        self.model = model

    def __call__(self, image_bgr: np.ndarray, threshold: float) -> sv.Detections:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.model.predict(rgb, threshold=threshold, include_source_image=False)


class _YOLOXBackend:
    """YOLOX ONNX export (the official release files) compiled by OpenVINO.

    Input: BGR image letterboxed into a square (top-left aligned, padded with
    grey 114), no normalisation. Output: one row per anchor point on the
    stride-8/16/32 grids with box offsets, objectness and 80 class scores, which
    ``decode_yolox`` turns into boxes before a per-class NMS.
    """

    def __init__(self, name: str, device: str = "CPU", nms: float = 0.45):
        import openvino as ov

        path = weights_dir() / "yolox" / f"{name}.onnx"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {name}.onnx (Apache-2.0, Megvii YOLOX release 0.1.1rc0) to {path}")
            urllib.request.urlretrieve(YOLOX_URL.format(name), path)
        core = ov.Core()
        model = core.read_model(path)
        self.size = model.inputs[0].partial_shape[2].get_length()
        if device not in core.available_devices:
            print(f"OpenVINO device {device} not available ({core.available_devices}); using CPU")
            device = "CPU"
        self.request = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()
        self.nms = nms
        self._grid, self._strides = yolox_grid(self.size)

    def __call__(self, image_bgr: np.ndarray, threshold: float) -> sv.Detections:
        h, w = image_bgr.shape[:2]
        ratio = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        canvas[:nh, :nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        raw = next(iter(self.request.infer({0: blob}).values()))[0]
        return decode_yolox(raw, self._grid, self._strides, ratio, threshold, self.nms, (w, h))


def yolox_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell coordinates and strides of the three YOLOX output grids, in output-row order."""
    grids, strides = [], []
    for s in (8, 16, 32):
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode_yolox(raw: np.ndarray, grid: np.ndarray, strides: np.ndarray, ratio: float, threshold: float,
                 nms: float, image_wh: tuple[int, int]) -> sv.Detections:
    """Grid outputs -> ``sv.Detections`` in the original image's pixels (score = objectness x class score)."""
    scores_all = raw[:, 4:5] * raw[:, 5:]
    class_id = scores_all.argmax(1)
    score = scores_all[np.arange(len(raw)), class_id]
    keep = score > threshold
    if not keep.any():
        return sv.Detections.empty()
    raw, class_id, score = raw[keep], class_id[keep], score[keep]
    xy = (raw[:, :2] + grid[keep]) * strides[keep]
    wh = np.exp(raw[:, 2:4]) * strides[keep]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / ratio
    w, h = image_wh
    xyxy = np.clip(xyxy, 0, [w, h, w, h])
    boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xyxy]
    idx = np.asarray(cv2.dnn.NMSBoxesBatched(boxes, score.tolist(), class_id.tolist(), threshold, nms), int).reshape(-1)
    return sv.Detections(
        xyxy=xyxy[idx].astype(np.float32),
        confidence=score[idx].astype(np.float32),
        class_id=class_id[idx].astype(int),
        data={"class_name": np.array([COCO80[c] for c in class_id[idx]], dtype=object)},
    )
