"""One video in, crossing events and counts out.

For each frame: detect (RF-DETR or YOLOX) -> track (ByteTrack from Roboflow's
``trackers``) -> reduce each track to an anchor point -> update the line
counter -> draw the overlay and write the frame. Frames are streamed, so memory
use does not grow with the length of the video.

Live use (a camera, or a file played at its real speed with ``realtime``): the
pipeline always works on the newest frame and drops the ones it had no time
for, so it never falls behind; it just looks at fewer frames per second. What
matters is that each object is still seen several times while it crosses.

Optional, for live use on small hardware:

* ``show``: open a window with the annotated frames as they are processed
  (``q`` or ``Esc`` stops the run and still writes the outputs);
* ``motion``: skip the detector on frames where nothing moves and nothing is
  being tracked (see ``linecount.motion``). A gate or a pedestrian door is still
  most of the time, so most frames never reach the model;
* ``snapshots``: save a photo of every crossing (always on for cameras).

Outputs in ``out_dir``, written as the run goes (a run cut by a power failure
keeps every crossing counted until then):

* ``events.csv``: one row per counted crossing (frame, time, track, class, direction,
  position, wall-clock time for live runs, snapshot file), flushed at each crossing;
* ``counts.csv``: in / out / total per class, updated at each crossing;
* ``snapshots/``: one JPEG per crossing;
* ``annotated.mp4``: the video with boxes, track IDs, classes, the line and the counters;
* ``tracks.csv``: every tracked box per frame, for files only (used to redraw figures);
* ``run.json``: the configuration and the measured speed, at the end.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from linecount.count import Line, LineCounter, anchor_points
from linecount.detect import VEHICLES, Detector, Roi
from linecount.draw import draw_frame, draw_status
from linecount.motion import MotionGate
from linecount.video import VideoWriter, is_live, probe, read_frames

WINDOW = "linecount  (q: stop)"


@dataclass
class RunConfig:
    line: str
    model: str = "nano"
    device: str = "CPU"  # OpenVINO device for YOLOX models: CPU or GPU (Intel graphics)
    threshold: float = 0.5
    classes: tuple[str, ...] = VEHICLES
    roi: str | None = None
    agnostic_nms: float | None = None
    anchor: str = "bottom_center"
    margin: float = 2.0
    min_frames: int = 1
    # ByteTrack (trackers.ByteTrackTracker); defaults are the library defaults
    track_activation_threshold: float = 0.7
    high_conf_det_threshold: float = 0.6
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold: float = 0.1
    lost_track_buffer: int = 30
    # video
    start: int = 0
    end: int | None = None
    stride: int = 1
    fps: float | None = None
    max_width: int = 1280
    write_video: bool = True
    title: str | None = None
    # live use
    show: bool = False
    motion: bool = False
    lang: str = "en"
    realtime: bool = False  # play a file at its own speed, dropping frames like a live camera
    snapshots: bool = False  # always on for live cameras


@dataclass
class RunResult:
    config: RunConfig
    frames: int
    seconds: float
    detector_seconds: float
    events: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    detector_frames: int = 0
    stopped: bool = False
    video_frames: int = 0  # frames of the source between the first and the last one processed

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


def run(video: str | Path, out_dir: str | Path, cfg: RunConfig, detector: Detector | None = None) -> RunResult:
    from trackers import ByteTrackTracker

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live = is_live(video)
    info = probe(video, cfg.max_width, cfg.fps)
    line = Line.parse(cfg.line)
    roi = Roi.parse(cfg.roi) if cfg.roi else None
    if detector is None:
        detector = Detector(cfg.model, cfg.threshold, tuple(cfg.classes), roi, cfg.agnostic_nms, cfg.device)
    effective_fps = info.fps / (1 if live or cfg.realtime else cfg.stride)
    tracker = ByteTrackTracker(
        lost_track_buffer=cfg.lost_track_buffer,
        frame_rate=effective_fps,
        track_activation_threshold=cfg.track_activation_threshold,
        minimum_consecutive_frames=cfg.minimum_consecutive_frames,
        minimum_iou_threshold=cfg.minimum_iou_threshold,
        high_conf_det_threshold=cfg.high_conf_det_threshold,
    )
    counter = LineCounter(line, margin=cfg.margin, min_frames=cfg.min_frames)
    gate = MotionGate(roi) if cfg.motion else None
    writer = None
    if cfg.write_video:
        # a live run keeps only the frames the model had time for, so its video plays at a nominal 8 fps
        writer = VideoWriter(out_dir / "annotated.mp4", info.width, info.height,
                             8.0 if live or cfg.realtime else effective_fps)

    wall_clock = live or cfg.realtime
    log = EventLog(out_dir, info.fps, snapshots=cfg.snapshots or live, line=line)
    # every box of every frame is only kept for files: on a camera running for days it would grow without end
    tracks_file = open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") if not live else None
    tracks_csv = csv.writer(tracks_file) if tracks_file else None
    if tracks_csv:
        tracks_csv.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
    frames = det_frames = 0
    last_seen = -(10**9)
    det_seconds = 0.0
    flash_until = -1
    stopped = False
    recent: list[float] = []
    first_index = last_index = None
    t0 = time.perf_counter()
    try:
        source = read_frames(video, cfg.max_width, cfg.start, cfg.end, cfg.stride, realtime=cfg.realtime, fps=cfg.fps)
        for index, frame in source:
            first_index = index if first_index is None else first_index
            last_index = index
            # the motion gate only skips the model when nothing moves AND nothing was detected recently,
            # so an object that stops on the line keeps being tracked
            moving = gate.moving(frame) if gate else True
            if moving or frames - last_seen < cfg.lost_track_buffer:
                td = time.perf_counter()
                det = detector(frame)
                det_seconds += time.perf_counter() - td
                det_frames += 1
                if len(det):
                    last_seen = frames
            else:
                det = sv.Detections.empty()
            tracked = tracker.update(det)
            tracked = tracked[tracked.tracker_id >= 0] if len(tracked) else tracked
            names = list(tracked.data.get("class_name", [])) if len(tracked) else []
            if len(tracked):
                pts = anchor_points(tracked.xyxy, cfg.anchor)
                new = counter.update(index, tracked.tracker_id, pts, names, tracked.confidence)
                if new:
                    flash_until = index + int(0.3 * info.fps)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if wall_clock else ""
                    boxes = dict(zip((int(t) for t in tracked.tracker_id), tracked.xyxy))
                    for e in new:
                        log.add(e, now, frame, boxes.get(e.track_id), counter.counts(), cfg.lang)
                names = [counter.track_class(int(t)) for t in tracked.tracker_id]  # voted class, steadier labels
                if tracks_csv:
                    for tid, name, conf, box in zip(tracked.tracker_id, names, tracked.confidence, tracked.xyxy):
                        tracks_csv.writerow([index, int(tid), name, f"{conf:.3f}", *(f"{v:.1f}" for v in box)])
            frames += 1
            if writer or cfg.show:
                img = draw_frame(
                    frame,
                    tracked.xyxy if len(tracked) else np.zeros((0, 4)),
                    tracked.tracker_id if len(tracked) else [],
                    names,
                    line,
                    counter.counts(),
                    tuple(cfg.classes),
                    flash=index <= flash_until,
                    counted_ids={e.track_id for e in counter.events},
                    title=cfg.title,
                    lang=cfg.lang,
                )
                if writer:
                    writer.write(img)
                if cfg.show:
                    recent = (recent + [time.perf_counter()])[-20:]
                    rate = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                    draw_status(img, rate, cfg.model, gate is not None and not moving, cfg.lang)
                    cv2.imshow(WINDOW, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        stopped = True
                        break
    except KeyboardInterrupt:
        stopped = True
    finally:
        if tracks_file:
            tracks_file.close()
        if writer:
            writer.close()
        if cfg.show:
            cv2.destroyAllWindows()
        log.close(counter.counts())
    seconds = time.perf_counter() - t0

    covered = last_index - first_index + 1 if first_index is not None else 0
    result = RunResult(cfg, frames, seconds, det_seconds, log.events, counter.counts(), det_frames, stopped, covered)
    _write_run_info(out_dir, result)
    return result


class EventLog:
    """Writes each crossing the moment it is counted, so nothing is lost if the run is cut.

    * ``events.csv`` gets one row per crossing, flushed to disk immediately;
    * ``counts.csv`` is rewritten with the running totals after every crossing;
    * with ``snapshots``, ``snapshots/`` gets a JPEG of the frame with that object's box.
    """

    COLUMNS = ["frame", "time_s", "track_id", "class", "direction", "position", "x", "y", "clock", "snapshot"]

    def __init__(self, out_dir: Path, fps: float, snapshots: bool = False, line: Line | None = None):
        self.out_dir = out_dir
        self.fps = fps
        self.line = line
        self.snap_dir = out_dir / "snapshots" if snapshots else None
        if self.snap_dir:
            self.snap_dir.mkdir(exist_ok=True)
        self._file = open(out_dir / "events.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(self.COLUMNS)
        self._file.flush()
        self.events: list[dict] = []
        self._write_counts({})

    def add(self, event, clock: str, frame: np.ndarray, box, counts: dict, lang: str = "en") -> None:
        snapshot = ""
        if self.snap_dir is not None:
            stamp = clock.replace("-", "").replace(":", "").replace(" ", "_") if clock else f"f{event.frame:06d}"
            snapshot = f"{stamp}_{event.track_id}_{event.class_name}_{event.direction}.jpg"
            cv2.imwrite(str(self.snap_dir / snapshot), _snapshot(frame, box, event, clock, lang, self.line),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        row = asdict(event) | {"clock": clock, "snapshot": snapshot}
        self.events.append(row)
        self._csv.writerow([event.frame, f"{event.frame / self.fps:.2f}", event.track_id, event.class_name,
                            event.direction, f"{event.position:.1f}", f"{event.x:.1f}", f"{event.y:.1f}",
                            clock, snapshot])
        self._file.flush()
        self._write_counts(counts)

    def _write_counts(self, counts: dict) -> None:
        tmp = self.out_dir / "counts.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "in", "out", "total"])
            for name, v in counts.items():
                w.writerow([name, v["in"], v["out"], v["in"] + v["out"]])
            w.writerow(["all", sum(v["in"] for v in counts.values()), sum(v["out"] for v in counts.values()),
                        sum(v["in"] + v["out"] for v in counts.values())])
        tmp.replace(self.out_dir / "counts.csv")  # atomic: a reader never sees a half-written file

    def close(self, counts: dict) -> None:
        self._write_counts(counts)
        self._file.close()


def _snapshot(frame: np.ndarray, box, event, clock: str, lang: str, line: Line | None = None) -> np.ndarray:
    from linecount.draw import color_for, tr

    img = frame.copy()
    if line is not None:
        cv2.line(img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (255, 255, 255), 2, cv2.LINE_AA)
    if box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_for(event.class_name), 3)
    label = f"{tr(event.class_name, lang)}  {tr(event.direction, lang)}  {clock or f'frame {event.frame}'}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    scale = min(1.0, 960 / img.shape[1])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img


def _write_run_info(out_dir: Path, result: RunResult) -> None:
    run_info = {
        "config": asdict(result.config),
        "frames_processed": result.frames,
        "frames_through_detector": result.detector_frames,
        "stopped_early": result.stopped,
        "wall_seconds": round(result.seconds, 2),
        "pipeline_fps": round(result.fps, 2),
        "video_frames_covered": result.video_frames,
        "share_of_frames_analyzed": round(result.frames / result.video_frames, 3) if result.video_frames else None,
        "video_frames_per_second": round(result.video_frames / result.seconds, 2) if result.seconds else 0.0,
        "detector_ms_per_frame": round(1000 * result.detector_seconds / max(1, result.detector_frames), 1),
        "total_count": len(result.events),
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
