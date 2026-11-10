"""Tests for the pieces added for live use: YOLOX decoding, the motion gate, live sources, labels and configs."""

import numpy as np

from linecount.detect import COCO80, decode_yolox, yolox_grid
from linecount.draw import tr
from linecount.evaluate import clip_run_config
from linecount.motion import MotionGate
from linecount.video import is_live


def _raw_with_one_box(size, row, cls, score=0.9):
    grid, strides = yolox_grid(size)
    raw = np.zeros((len(grid), 85), np.float32)
    raw[:, 4] = 0.01  # background objectness
    raw[row, :4] = [0.5, 0.5, np.log(4), np.log(2)]  # centre of the cell, 4x2 strides in size
    raw[row, 4] = 1.0
    raw[row, 5 + cls] = score
    return raw, grid, strides


def test_yolox_grid_shape():
    grid, strides = yolox_grid(416)
    assert len(grid) == 52 * 52 + 26 * 26 + 13 * 13 == 3549
    assert strides[0, 0] == 8 and strides[-1, 0] == 32


def test_decode_yolox_box_in_original_pixels():
    size = 416
    row = 52 * 10 + 20  # stride-8 grid, cell x=20, y=10
    raw, grid, strides = _raw_with_one_box(size, row, COCO80.index("car"))
    ratio = 0.5  # the image was shrunk to half before the model
    det = decode_yolox(raw, grid, strides, ratio, 0.3, 0.45, (832, 832))
    assert len(det) == 1
    assert det.data["class_name"][0] == "car"
    cx, cy = (20.5 * 8) / ratio, (10.5 * 8) / ratio
    w, h = 32 / ratio, 16 / ratio
    np.testing.assert_allclose(det.xyxy[0], [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], atol=1e-3)
    assert abs(det.confidence[0] - 0.9) < 1e-6


def test_decode_yolox_threshold_and_empty():
    raw, grid, strides = _raw_with_one_box(416, 5, COCO80.index("person"), score=0.2)
    assert len(decode_yolox(raw, grid, strides, 1.0, 0.3, 0.45, (416, 416))) == 0


def test_motion_gate():
    gate = MotionGate()
    still = np.full((360, 640, 3), 120, np.uint8)
    assert gate.moving(still)  # first frame always runs the model
    assert not gate.moving(still.copy())
    moved = still.copy()
    moved[100:200, 100:200] = 255
    assert gate.moving(moved)


def test_is_live():
    assert is_live(0) and is_live("1") and is_live("rtsp://10.0.0.5:554/stream")
    assert not is_live("clips/a40.webm")


def test_spanish_labels():
    assert tr("car", "es") == "auto" and tr("in", "es") == "entran"
    assert tr("car", "en") == "car" and tr("unknown thing", "es") == "unknown thing"


def test_fast_config_is_yolox():
    cfg = clip_run_config("fast", "a40")
    assert cfg.model == "yolox-tiny" and cfg.threshold == 0.3 and cfg.roi
    assert clip_run_config("final", "roundabout").roi  # "auto" resolves to the clip's crop
