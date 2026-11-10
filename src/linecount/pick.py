"""Draw the counting line (and optionally a detection crop) with the mouse on a frame of the video.

    uv run python -m linecount.pick video.mp4
    uv run python -m linecount.pick 0            # webcam

Click two points for the line. Then press Enter to accept, or drag a rectangle
first with the right mouse button to set a crop (``--roi``) around the line.
``r`` starts over, ``Esc`` cancels. The coordinates are printed in the same
pixels the counter uses (after the frame is scaled to ``--max-width``).
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from linecount.video import is_live, read_frames

WINDOW = "linecount: click 2 points for the line (right-drag: crop, Enter: ok, r: reset, Esc: cancel)"


def first_frame(source, max_width: int = 1280, skip: int = 0) -> np.ndarray:
    """Frame ``skip`` of a file, or a frame a moment after a camera opens (the first ones are often dark)."""
    frames = read_frames(source, max_width, end=10) if is_live(source) else \
        read_frames(source, max_width, start=skip, end=skip + 1)
    frame = None
    for _, frame in frames:
        pass
    if frame is None:
        raise SystemExit(f"could not read a frame from {source}")
    return frame


def pick(frame: np.ndarray) -> tuple[str | None, str | None]:
    """Interactive picker. Returns ``(line, roi)`` as ``"x1,y1,x2,y2"`` strings (``None`` when not set)."""
    points: list[tuple[int, int]] = []
    roi: list[int] = []
    drag: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            drag[:] = [(x, y), (x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drag and flags & cv2.EVENT_FLAG_RBUTTON:
            drag[1] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and drag:
            (ax, ay), (bx, by) = drag[0], (x, y)
            if abs(bx - ax) > 10 and abs(by - ay) > 10:
                roi[:] = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
            drag.clear()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            img = frame.copy()
            if roi:
                cv2.rectangle(img, tuple(roi[:2]), tuple(roi[2:]), (0, 200, 255), 2)
            if len(drag) == 2:
                cv2.rectangle(img, drag[0], drag[1], (0, 200, 255), 1)
            for p in points:
                cv2.circle(img, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
            if len(points) == 2:
                cv2.line(img, points[0], points[1], (0, 0, 0), 6, cv2.LINE_AA)
                cv2.line(img, points[0], points[1], (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(points) == 2:
                break
            if key == ord("r"):
                points.clear(), roi.clear()
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None, None
    finally:
        cv2.destroyAllWindows()
    line = ",".join(str(v) for p in points for v in p)
    return line, ",".join(map(str, roi)) if roi else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file, webcam number (0) or camera URL (rtsp://...)")
    ap.add_argument("--frame", type=int, default=0, help="frame of the video to draw on")
    ap.add_argument("--max-width", type=int, default=1280)
    args = ap.parse_args(argv)
    line, roi = pick(first_frame(args.video, args.max_width, args.frame))
    if line is None:
        print("cancelled", file=sys.stderr)
        return 1
    print(f"--line {line}" + (f" --roi {roi}" if roi else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
