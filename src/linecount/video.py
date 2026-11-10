"""Video input and output as a stream of frames.

Frames are decoded one at a time and never held in memory as a whole video, so
long or high-resolution files only cost one frame of RAM. Frames larger than
``max_width`` are downscaled on the fly, and the frame rate can be overridden
for files whose container reports a wrong value (some WebM files on Wikimedia
Commons report 1000 fps).

The source can also be live: a webcam number (``0``) or a camera URL such as
``rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101``. A live camera does
not wait for a slow model, so live frames are read on a background thread and
the pipeline always takes the newest one; frames that arrive while the model is
busy are dropped instead of piling up as delay. A camera that stops sending is
reopened until it comes back. A video file can be played the same way, at its
own speed (``realtime=True``), to test the counter under live conditions.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_live(source: str | Path | int) -> bool:
    """True for a webcam number or a camera/stream URL, False for a video file."""
    if isinstance(source, int):
        return True
    text = str(source).strip()
    return text.isdigit() or text.lower().startswith(STREAM_PREFIXES)


def open_capture(source: str | Path | int) -> cv2.VideoCapture:
    if isinstance(source, int) or str(source).strip().isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(str(source))


@dataclass(frozen=True)
class VideoInfo:
    width: int  # after resizing
    height: int
    fps: float
    source_width: int
    source_height: int

    @property
    def scale(self) -> float:
        return self.width / self.source_width


def probe(path: str | Path | int, max_width: int = 1280, fps: float | None = None) -> VideoInfo:
    cap = open_capture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported = cap.get(cv2.CAP_PROP_FPS)
    if not sw or not sh:  # some streams only report their size after the first frame
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise FileNotFoundError(f"cannot read a frame from: {path}")
        sh, sw = frame.shape[:2]
    cap.release()
    if is_live(path) and not 0 < reported <= 120:
        reported = 25.0  # webcams and some cameras report 0 or nonsense
    scale = min(1.0, max_width / sw)
    w, h = round(sw * scale), round(sh * scale)
    return VideoInfo(w - w % 2, h - h % 2, fps or reported or 30.0, sw, sh)


def read_frames(
    path: str | Path | int,
    max_width: int = 1280,
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
    realtime: bool = False,
    fps: float | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` for frames in ``[start, end)`` every ``stride`` frames.

    Frame indices always refer to the original video, so a crossing logged at
    frame 412 is frame 412 whatever the stride. For a live source the newest
    frame is always taken (``start`` and ``stride`` do not apply), indices count
    the frames the camera delivered, and ``end`` caps that count.

    ``realtime=True`` plays a video file as if it were a camera: frames become
    available at the video's own pace (``fps``), and a consumer slower than that
    only gets the newest one, exactly as with a live source. This is how the
    counter is tested under real-time conditions against the manual counts.
    """
    info = probe(path, max_width, fps)
    if is_live(path):
        yield from _read_newest(_LatestFrame(path, reconnect=True), info, end)
        return
    if realtime:
        yield from _read_newest(_LatestFrame(path, pace_fps=info.fps, start=start, end=end), info, end)
        return
    cap = open_capture(path)
    index = 0
    try:
        while end is None or index < end:
            if index < start or (index - start) % stride:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield index, _fit(frame, info)
            index += 1
    finally:
        cap.release()


def _fit(frame: np.ndarray, info: VideoInfo) -> np.ndarray:
    if frame.shape[1] != info.width or frame.shape[0] != info.height:
        frame = cv2.resize(frame, (info.width, info.height), interpolation=cv2.INTER_AREA)
    return frame


class _LatestFrame:
    """Reads a source on a background thread and keeps only the newest frame.

    * Live camera (``reconnect=True``): if frames stop for about 5 s, the camera
      is reopened every few seconds until it answers again; the run never ends
      on its own.
    * Video file played as a camera (``pace_fps``): frame ``i`` is published at
      ``(i - start) / pace_fps`` seconds after the reader starts, then the file ends.
    """

    def __init__(self, source: str | Path | int, reconnect: bool = False, pace_fps: float | None = None,
                 start: int = 0, end: int | None = None):
        self._source = source
        self._cap = open_capture(source)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {source}")
        self._reconnect = reconnect
        self._pace = pace_fps
        self._start, self._end = start, end
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._index = -1
        self._running = True
        self.reconnections = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _publish(self, index: int, frame: np.ndarray) -> None:
        with self._cond:
            self._frame, self._index = frame, index
            self._cond.notify_all()

    def _loop(self) -> None:
        try:
            if self._pace:
                self._loop_paced()
            else:
                self._loop_live()
        finally:
            with self._cond:
                self._running = False
                self._cond.notify_all()

    def _loop_live(self) -> None:
        failures, index = 0, -1
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                failures, index = 0, index + 1
                self._publish(index, frame)
                continue
            failures += 1
            if failures <= 50:  # up to about 5 s of hiccups
                time.sleep(0.1)
                continue
            if not self._reconnect:
                return
            print(f"camera {self._source} stopped sending frames; reconnecting...", flush=True)
            self._cap.release()
            time.sleep(3)
            self._cap = open_capture(self._source)
            self.reconnections += 1
            failures = 0

    def _loop_paced(self) -> None:
        index = 0
        while index < self._start and self._cap.grab():
            index += 1
        t0 = time.perf_counter()
        while self._running and (self._end is None or index < self._end):
            ok, frame = self._cap.read()
            if not ok:
                return
            wait = t0 + (index - self._start) / self._pace - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            self._publish(index, frame)
            index += 1

    def next(self, after: int) -> tuple[int, np.ndarray] | None:
        """Wait for a frame newer than ``after``; ``None`` once the source has ended.

        A reconnecting camera never ends, so this keeps waiting while it reconnects.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._index > after or not self._running)
            if self._index <= after:
                return None
            return self._index, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=5)
        self._cap.release()


def _read_newest(reader: _LatestFrame, info: VideoInfo, end: int | None) -> Iterator[tuple[int, np.ndarray]]:
    last = -1
    try:
        while end is None or last < end - 1:
            item = reader.next(last)
            if item is None:
                break
            last, frame = item
            yield last, _fit(frame, info)
    finally:
        reader.close()


class VideoWriter:
    """Thin wrapper over ``cv2.VideoWriter`` (MPEG-4 Part 2, plays in VLC and most desktop players)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, codec: str = "mp4v"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path} with codec {codec}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
