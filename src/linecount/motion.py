"""A cheap motion check that decides when the detector needs to run.

A camera at a gate or a door sees nothing happen most of the time. Comparing
each frame with the previous one on a small grey copy costs well under a
millisecond, while the detector costs 40 to 500 ms. So the pipeline skips the
detector when this gate says nothing moved and nothing was detected recently.

Changed pixels are counted only inside the ``roi`` when one is given, so trees
moving outside the counting area do not wake the model up.
"""

from __future__ import annotations

import cv2
import numpy as np

from linecount.detect import Roi


class MotionGate:
    def __init__(self, roi: Roi | None = None, width: int = 320, pixel_delta: int = 25, min_fraction: float = 0.002):
        self.roi = roi
        self.width = width
        self.pixel_delta = pixel_delta
        self.min_fraction = min_fraction
        self._prev: np.ndarray | None = None

    def _small(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is not None:
            r = self.roi.clip(frame.shape[1], frame.shape[0])
            frame = frame[r.y1 : r.y2, r.x1 : r.x2]
        scale = min(1.0, self.width / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grey, (5, 5), 0)

    def moving(self, frame: np.ndarray) -> bool:
        """True when enough pixels changed since the previous frame (always True on the first frame)."""
        small = self._small(frame)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return True
        changed = cv2.absdiff(small, prev) > self.pixel_delta
        return float(changed.mean()) >= self.min_fraction
