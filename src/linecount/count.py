"""Line-crossing counter for tracked objects.

Each tracked object is reduced to one anchor point per frame (the bottom-centre
of its box for vehicles, the centre for people). The counter remembers on which
side of the line each track was last seen and logs an event when the track shows
up on the other side.

Two rules keep the count stable when boxes jitter:

* a side only counts once the anchor is at least ``margin`` pixels away from the
  line, so a box wobbling on the line does not flip back and forth;
* by default a track is counted at most once, so a track that crosses and comes
  back is not counted twice.

The class of an event is the confidence-weighted vote over every frame of the
track up to the crossing, so a car that is labelled ``truck`` for two frames is
still counted as a car.

This plays the same role as ``supervision.LineZone``, plus the per-event log
(frame, position along the line, class, direction) needed to evaluate the count
event by event. It has no dependency on the detector, so it is unit-tested on
synthetic tracks.

Direction convention: ``in`` means moving to the right-hand side of the line as
drawn from (x1, y1) to (x2, y2) on screen. For a horizontal line drawn left to
right, ``in`` is moving down the image; ``out`` is moving up.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def parse(cls, text: str) -> Line:
        parts = [float(v) for v in text.replace(" ", "").split(",")]
        if len(parts) != 4:
            raise ValueError(f"line must be x1,y1,x2,y2, got {text!r}")
        line = cls(*parts)
        if line.length == 0:
            raise ValueError("line endpoints must differ")
        return line

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    def scaled(self, factor: float) -> Line:
        return Line(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Perpendicular distance in pixels; positive on the right-hand side (see module docstring)."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        cross = dx * (points[:, 1] - self.y1) - dy * (points[:, 0] - self.x1)
        return cross / self.length

    def position(self, points: np.ndarray) -> np.ndarray:
        """Distance along the line from (x1, y1), in pixels, of each point's projection."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        return ((points[:, 0] - self.x1) * dx + (points[:, 1] - self.y1) * dy) / self.length


def anchor_points(xyxy: np.ndarray, anchor: str = "bottom_center") -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    if anchor == "bottom_center":
        return np.stack([cx, xyxy[:, 3]], axis=1)
    if anchor == "center":
        return np.stack([cx, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
    raise ValueError(f"unknown anchor {anchor!r}")


@dataclass
class CrossingEvent:
    frame: int
    track_id: int
    class_name: str
    direction: str  # "in" or "out"
    position: float  # pixels along the line from (x1, y1)
    x: float
    y: float


@dataclass
class _TrackState:
    side: int = 0  # last confirmed side: +1, -1, or 0 if never confirmed
    frames: int = 0
    counted: bool = False
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class LineCounter:
    def __init__(self, line: Line, margin: float = 4.0, once_per_track: bool = True, min_frames: int = 1):
        self.line = line
        self.margin = margin
        self.once_per_track = once_per_track
        self.min_frames = min_frames
        self.events: list[CrossingEvent] = []
        self._tracks: dict[int, _TrackState] = defaultdict(_TrackState)

    def update(
        self,
        frame: int,
        track_ids: np.ndarray,
        points: np.ndarray,
        class_names: list[str] | np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> list[CrossingEvent]:
        """Feed one frame of tracked anchors; return the events triggered in this frame."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if confidences is None:
            confidences = np.ones(len(points))
        dist = self.line.signed_distance(points)
        pos = self.line.position(points)
        new: list[CrossingEvent] = []
        for tid, p, d, s, name, conf in zip(track_ids, points, dist, pos, class_names, confidences):
            state = self._tracks[int(tid)]
            state.frames += 1
            state.votes[str(name)] += float(conf)
            if abs(d) < self.margin or not (0.0 <= s <= self.line.length):
                continue  # in the dead zone around the line, or beside the segment
            side = 1 if d > 0 else -1
            if state.side != 0 and side != state.side and state.frames >= self.min_frames:
                if not (self.once_per_track and state.counted):
                    event = CrossingEvent(
                        frame=frame,
                        track_id=int(tid),
                        class_name=max(state.votes, key=state.votes.get),
                        direction="in" if side > 0 else "out",
                        position=float(s),
                        x=float(p[0]),
                        y=float(p[1]),
                    )
                    new.append(event)
                    state.counted = True
            state.side = side
        self.events.extend(new)
        return new

    def track_class(self, track_id: int) -> str:
        """Current confidence-weighted class vote of a track."""
        votes = self._tracks[int(track_id)].votes
        return max(votes, key=votes.get) if votes else ""

    def counts(self) -> dict[str, dict[str, int]]:
        """``{class_name: {"in": n, "out": n}}`` over all events so far."""
        out: dict[str, dict[str, int]] = {}
        for e in self.events:
            out.setdefault(e.class_name, {"in": 0, "out": 0})[e.direction] += 1
        return dict(sorted(out.items()))

    @property
    def total(self) -> int:
        return len(self.events)
