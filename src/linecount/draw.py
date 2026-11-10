"""Overlay drawing: boxes with track IDs and class names, the counting line and a live per-class counter panel.

Plain OpenCV drawing, so the look is the same in the output video, the README
figures and the web assets.
"""

from __future__ import annotations

import cv2
import numpy as np

from linecount.count import Line

# BGR colours per class; anything else is grey.
CLASS_COLORS = {  # validated categorical palette (colour-blind safe), as BGR
    "car": (214, 120, 42),  # #2a78d6 blue
    "truck": (52, 104, 235),  # #eb6834 orange
    "bus": (122, 175, 27),  # #1baf7a aqua
    "motorcycle": (164, 123, 232),  # #e87ba4 magenta
    "person": (0, 161, 237),  # #eda100 yellow
    "bicycle": (214, 80, 98),  # #6250d6 violet
}
DEFAULT_COLOR = (180, 180, 180)
LINE_COLOR = (255, 255, 255)
FLASH_COLOR = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# On-screen words. OpenCV's built-in fonts are ASCII only, so Spanish labels go without accents.
LABELS = {
    "en": {"in": "in", "out": "out", "total": "total", "all": "all"},
    "es": {"in": "entran", "out": "salen", "total": "total", "all": "todos",
           "car": "auto", "truck": "camion", "bus": "colectivo", "motorcycle": "moto",
           "person": "persona", "bicycle": "bici"},
}


def tr(word: str, lang: str = "en") -> str:
    return LABELS.get(lang, {}).get(word, word)


def color_for(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, DEFAULT_COLOR)


def _label(img, text, x, y, color, scale):
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(th + base + 2, y)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 1), FONT, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    xyxy: np.ndarray,
    track_ids: np.ndarray,
    class_names,
    line: Line,
    counts: dict[str, dict[str, int]],
    classes: tuple[str, ...],
    flash: bool = False,
    counted_ids: set[int] | None = None,
    title: str | None = None,
    lang: str = "en",
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.4, w / 1280 * 0.5)
    thick = max(1, int(round(w / 640)))
    counted_ids = counted_ids or set()

    min_label_h = 0.03 * h  # tiny far-away boxes get no text, to keep the frame readable
    for box, tid, name in zip(np.asarray(xyxy).reshape(-1, 4), track_ids, class_names):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = color_for(str(name))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + (1 if int(tid) in counted_ids else 0))
        if y2 - y1 >= min_label_h:
            _label(img, f"#{int(tid)} {tr(str(name), lang)}", x1, y1, color, scale)

    lc = FLASH_COLOR if flash else LINE_COLOR
    p1, p2 = (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2))
    cv2.line(img, p1, p2, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.line(img, p1, p2, lc, thick + 2, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, thick + 4, lc, -1, cv2.LINE_AA)

    # counter panel, top-left
    rows = [(c, counts.get(c, {"in": 0, "out": 0})) for c in classes]
    rows += [(c, counts[c]) for c in sorted(set(counts) - set(classes))]
    total_in = sum(v["in"] for _, v in rows)
    total_out = sum(v["out"] for _, v in rows)
    k = scale / 0.5
    line_h, pad = int(30 * k), int(12 * k)
    widths = (175, 235, 305) if lang == "en" else (200, 280, 350)  # Spanish headers are longer
    cols = [int(v * k) for v in widths]  # right edges of the in / out / total columns
    panel_w = cols[-1] + pad
    panel_h = pad + line_h * (len(rows) + 2 + (1 if title else 0)) + int(4 * k)
    # put the panel in the corner farthest from the line, so it never hides the counting area
    mx, my = (line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2
    corners = [(0, 0), (w - panel_w, 0), (0, h - panel_h), (w - panel_w, h - panel_h)]
    ox, oy = max(corners, key=lambda c: (c[0] + panel_w / 2 - mx) ** 2 + (c[1] + panel_h / 2 - my) ** 2 - 1e-3 * (c[0] + c[1]))
    overlay = img.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + panel_w, oy + panel_h), (25, 25, 25), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    tt = max(1, int(round(scale * 2)))

    def put(text, x, y, color, right=False):
        if right:
            x -= cv2.getTextSize(text, FONT, scale, tt)[0][0]
        cv2.putText(img, text, (ox + x, oy + y), FONT, scale, color, tt, cv2.LINE_AA)

    y = pad + int(20 * k)
    if title:
        put(title, pad, y, (235, 235, 235))
        y += line_h
    for text, x in zip(("in", "out", "total"), cols):
        put(tr(text, lang), x, y, (170, 170, 170), right=True)
    y += line_h
    for name, v in rows:
        cv2.circle(img, (ox + pad + int(6 * k), oy + y - int(7 * k)), int(6 * k), color_for(name), -1, cv2.LINE_AA)
        put(tr(name, lang), pad + int(20 * k), y, (240, 240, 240))
        for text, x in zip((v["in"], v["out"], v["in"] + v["out"]), cols):
            put(str(text), x, y, (240, 240, 240), right=True)
        y += line_h
    put(tr("all", lang), pad + int(20 * k), y, LINE_COLOR)
    for text, x in zip((total_in, total_out, total_in + total_out), cols):
        put(str(text), x, y, LINE_COLOR, right=True)
    return img


def draw_status(img: np.ndarray, fps: float, model: str, idle: bool = False, lang: str = "en") -> None:
    """Small strip at the bottom centre with the live processing speed (only in the preview window)."""
    h, w = img.shape[:2]
    scale = max(0.4, w / 1280 * 0.45)
    tt = max(1, int(round(scale * 2)))
    text = f"{fps:4.1f} fps  |  {model}"
    if idle:
        text += "  |  " + ("sin movimiento: modelo en pausa" if lang == "es" else "no motion: model paused")
    (tw, th), base = cv2.getTextSize(text, FONT, scale, tt)
    x, y = (w - tw) // 2, h - 10
    cv2.rectangle(img, (x - 8, y - th - base - 6), (x + tw + 8, y + 4), (25, 25, 25), -1)
    cv2.putText(img, text, (x, y - base + 2), FONT, scale, (235, 235, 235), tt, cv2.LINE_AA)
