"""Figures for the README and the website, redrawn from a saved run.

The run folder keeps every tracked box per frame (``tracks.csv``) and the
results folder keeps the counted events, so the overlay can be redrawn on the
original frames at any size without running the model again:

* ``figures/hero.jpg``: one annotated frame, 1600 px wide;
* ``figures/demo.gif``: a short loop for the README;
* ``figures/web/``: an H.264 MP4 clip, a GIF and a 1600 px still for the website;
* ``figures/evaluation.png``: true vs predicted counts per clip;
* ``figures/errors_*.jpg``: frames that show what fails.

    uv run --group web python -m linecount.figures --data <clips folder> --runs <runs folder>

The H.264 clip needs PyAV (``web`` dependency group). It uses the Windows Media
Foundation encoder when present and falls back to libx264 elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from linecount.count import Line
from linecount.detect import PEOPLE, VEHICLES
from linecount.draw import draw_frame
from linecount.evaluate import RESULTS_DIR, ROOT, load_json
from linecount.video import probe, read_frames

FIG = ROOT / "figures"


class Run:
    """A saved run of one clip: source video, tracks and counted events."""

    def __init__(self, clip_name: str, config: str, data_dir: Path, runs_dir: Path):
        self.name = clip_name
        self.clip = load_json("clips.json")[clip_name]
        self.source = data_dir / self.clip["file"]
        self.fps = self.clip.get("fps") or probe(self.source).fps
        self.proc = probe(self.source, 1280, self.fps)  # coordinates in tracks/events are in this frame size
        self.line = Line.parse(self.clip["line"])
        self.classes = VEHICLES if self.clip["classes"] == "vehicles" else PEOPLE
        self.tracks: dict[int, list] = defaultdict(list)
        with open(runs_dir / config / clip_name / "tracks.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                self.tracks[int(r["frame"])].append((int(r["track_id"]), r["class"], box))
        with open(RESULTS_DIR / config / f"{clip_name}_events.csv", newline="", encoding="utf-8") as f:
            self.events = [(int(r["frame"]), int(r["track_id"]), r["class"], r["direction"]) for r in csv.DictReader(f)]

    def counts_at(self, frame: int) -> dict:
        out: dict[str, dict[str, int]] = {}
        for f, _, name, direction in self.events:
            if f <= frame:
                out.setdefault(name, {"in": 0, "out": 0})[direction] += 1
        return out

    def render(self, start: int, end: int, width: int, step: int = 1):
        """Yield ``(frame_index, bgr_image)`` with the overlay drawn at ``width`` pixels."""
        scale = width / self.proc.width
        height = int(round(self.proc.height * scale / 2)) * 2
        line = self.line.scaled(scale)
        flash_frames = int(0.3 * self.fps)
        for index, frame in read_frames(self.source, 4096, start, end, step):
            interp = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_CUBIC
            frame = cv2.resize(frame, (width, height), interpolation=interp)
            rows = self.tracks.get(index, [])
            boxes = np.array([r[2] for r in rows]).reshape(-1, 4) * scale
            counted = {t for f, t, _, _ in self.events if f <= index}
            flash = any(index - flash_frames < f <= index for f, _, _, _ in self.events)
            img = draw_frame(frame, boxes, [r[0] for r in rows], [r[1] for r in rows], line,
                             self.counts_at(index), self.classes, flash=flash, counted_ids=counted)
            yield index, img


def save_jpg(path: Path, img: np.ndarray, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.2f} MB")


def still(run: Run, frame: int, path: Path, width: int = 1600) -> None:
    for _, img in run.render(frame, frame + 1, width):
        save_jpg(path, img, 92)


def gif(run: Run, start: int, end: int, path: Path, width: int = 640, fps: float = 10, max_mb: float = 5.0,
        colors: int = 96) -> None:
    from PIL import Image

    step = max(1, round(run.fps / fps))
    frames = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for _, img in run.render(start, end, width, step)]
    pal = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 * step / run.fps), loop=0,
                optimize=True, disposal=1)
    size = path.stat().st_size / 1e6
    print(f"{path.relative_to(ROOT)}  {size:.2f} MB, {len(frames)} frames, {width}px")
    if size > max_mb:
        raise SystemExit(f"{path} is {size:.1f} MB, above the {max_mb} MB limit")


def mp4_h264(run: Run, start: int, end: int, path: Path, width: int = 1280, bitrate: int = 2_400_000) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(path), "w", options={"movflags": "faststart"})
    codec = "h264_mf" if "h264_mf" in av.codecs_available else "libx264"
    stream = out.add_stream(codec, rate=round(run.fps * 1000) / 1000 if run.fps % 1 else int(run.fps))
    stream.width = width
    stream.height = int(round(run.proc.height * width / run.proc.width / 2)) * 2
    stream.pix_fmt = "nv12" if codec == "h264_mf" else "yuv420p"
    stream.bit_rate = bitrate
    n = 0
    for _, img in run.render(start, end, width):
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            out.mux(packet)
        n += 1
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.2f} MB, {n} frames, {n / run.fps:.1f} s, {codec}")


def evaluation_chart(rows_spec: list[dict], path: Path) -> None:
    """Horizontal bars per clip: true crossings vs crossings counted correctly (+ wrong extra counts)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, true, correct, extra = [], [], [], []
    for r in rows_spec:
        m = json.loads((RESULTS_DIR / r["config"] / "metrics.json").read_text(encoding="utf-8"))[r["clip"]]
        s = m[r.get("split", "all")]
        labels.append(r["label"])
        true.append(s["true_count"])
        correct.append(s["correct"])
        extra.append(s["double_counts"] + s["false_crossings"])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.95 * len(labels)), dpi=150)
    ax.barh(y - 0.2, true, height=0.36, color="#b9b8b3", label="true crossings (manual count)")
    ax.barh(y + 0.2, correct, height=0.36, color="#2a78d6", label="counted correctly")
    ax.barh(y + 0.2, extra, left=correct, height=0.36, color="#eb6834", label="extra counts (double or false)")
    for i in range(len(labels)):
        ax.text(true[i] + 0.3, y[i] - 0.2, str(true[i]), va="center", fontsize=9, color="#52514e")
        ax.text(correct[i] + extra[i] + 0.3, y[i] + 0.2, f"{correct[i]}" + (f" + {extra[i]}" if extra[i] else ""),
                va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("crossings of the counting line")
    ax.set_xlim(0, max(true + [c + e for c, e in zip(correct, extra)]) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.relative_to(ROOT)}")


def evidence(runs: list[Run], frames: list[int], crop: tuple[int, int, int, int], path: Path, captions: list[str],
             tile_width: int = 640) -> None:
    """Side-by-side crops of annotated frames; ``crop`` is x1,y1,x2,y2 in the 1280-wide frame."""
    tiles = []
    for run, f, cap in zip(runs, frames, captions):
        for _, img in run.render(f, f + 1, run.proc.width):
            x1, y1, x2, y2 = crop
            t = img[y1:y2, x1:x2]
            t = cv2.resize(t, (tile_width, int(t.shape[0] * tile_width / t.shape[1])), interpolation=cv2.INTER_CUBIC)
            bar = np.full((34, tile_width, 3), 245, np.uint8)
            cv2.putText(bar, cap, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, t]))
    sep = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.hstack([out, sep, t])
    save_jpg(path, out, 90)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", default="final")
    ap.add_argument("--only", default=None, help="comma separated: hero,gif,web,chart,errors")
    args = ap.parse_args(argv)
    data, runs = Path(args.data), Path(args.runs)
    only = set(args.only.split(",")) if args.only else {"hero", "gif", "web", "chart", "errors"}
    spec = json.loads((ROOT / "evaluation" / "figures.json").read_text(encoding="utf-8"))
    cache: dict[str, Run] = {}

    def run_for(name, config=None):
        key = f"{config or args.config}/{name}"
        if key not in cache:
            cache[key] = Run(name, config or args.config, data, runs)
        return cache[key]

    if "hero" in only:
        h = spec["hero"]
        still(run_for(h["clip"]), h["frame"], FIG / "hero.jpg", 1600)
        still(run_for(h["clip"]), h["frame"], FIG / "web" / "still_1600.jpg", 1600)
    if "gif" in only:
        g = spec["gif"]
        gif(run_for(g["clip"]), g["start"], g["end"], FIG / "demo.gif", g.get("width", 640), g.get("fps", 10), 8.0,
            g.get("colors", 96))
    if "web" in only:
        w = spec["web"]
        mp4_h264(run_for(w["clip"]), w["start"], w["end"], FIG / "web" / "demo_h264.mp4", w.get("width", 1280))
        gif(run_for(w["clip"]), w.get("gif_start", w["start"]), w.get("gif_end", w["end"]), FIG / "web" / "demo.gif",
            w.get("gif_width", 640), w.get("gif_fps", 10), 5.0, w.get("gif_colors", 96))
    if "chart" in only:
        evaluation_chart(spec["chart"], FIG / "evaluation.png")
    if "errors" in only:
        for e in spec.get("errors", []):
            configs = e.get("configs", [args.config] * len(e["frames"]))
            tile_runs = [run_for(e["clip"], c) for c in configs]
            evidence(tile_runs, e["frames"], tuple(e["crop"]), FIG / e["file"], e["captions"], e.get("tile_width", 640))


if __name__ == "__main__":
    main()
