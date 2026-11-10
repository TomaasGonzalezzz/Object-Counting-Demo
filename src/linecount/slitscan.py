"""Tools for building the manual ground truth.

A *slit-scan* image samples the pixels under the counting line in every frame
and stacks them top to bottom: the horizontal axis is the position along the
line and the vertical axis is time (frame number). Every object that crosses the
line leaves a blob, so the crossings of a whole clip can be read and logged by
frame and position from a handful of images instead of stepping through
hundreds of frames. Blobs that are ambiguous (two cars close together, a person
standing on the line) are then checked on the actual frames with ``--crops``,
which tiles the region around the line for a range of frames.

    uv run python -m linecount.slitscan VIDEO --line x1,y1,x2,y2 --out DIR
    uv run python -m linecount.slitscan VIDEO --line ... --out DIR --crops 300 360 3

The images are only an aid for a person; nothing here decides what counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from linecount.count import Line
from linecount.video import probe, read_frames


def _sample_maps(line: Line) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(line.length))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    map_x = (line.x1 + t * (line.x2 - line.x1)).astype(np.float32)[None, :]
    map_y = (line.y1 + t * (line.y2 - line.y1)).astype(np.float32)[None, :]
    return map_x, map_y


def slit_scan(path, line: Line, start=0, end=None, max_width=1280) -> np.ndarray:
    map_x, map_y = _sample_maps(line)
    rows = [cv2.remap(f, map_x, map_y, cv2.INTER_LINEAR) for _, f in read_frames(path, max_width, start, end)]
    return np.vstack(rows)


def render_chunks(scan: np.ndarray, out_dir: Path, start: int, chunk: int = 150, tscale: int = 3, prefix="slit"):
    """Write the slit-scan as labelled PNGs: frame numbers on the left, position along the line on top."""
    out_dir.mkdir(parents=True, exist_ok=True)
    margin_l, margin_t = 60, 22
    paths = []
    for c0 in range(0, len(scan), chunk):
        part = np.repeat(scan[c0 : c0 + chunk], tscale, axis=0)
        h, w = part.shape[:2]
        canvas = np.full((h + margin_t, w + margin_l, 3), 255, np.uint8)
        canvas[margin_t:, margin_l:] = part
        for x in range(0, w, 50):
            cv2.line(canvas, (margin_l + x, margin_t - 6), (margin_l + x, margin_t), (0, 0, 0), 1)
            cv2.putText(canvas, str(x), (margin_l + x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for r in range(0, len(scan[c0 : c0 + chunk])):
            frame = start + c0 + r
            if frame % 5 == 0:
                y = margin_t + r * tscale
                cv2.line(canvas, (margin_l - (14 if frame % 10 == 0 else 6), y), (margin_l, y), (0, 0, 0), 1)
                if frame % 10 == 0:
                    cv2.putText(canvas, str(frame), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        p = out_dir / f"{prefix}_{start + c0:05d}.png"
        cv2.imwrite(str(p), canvas)
        paths.append(p)
    return paths


def crop_tiles(path, line: Line, frames: range, out: Path, pad: int = 60, cols: int = 4, max_width=1280, tile_w=480):
    """Tile the region around the line for the given frames, each labelled with its frame number."""
    x0 = int(max(0, min(line.x1, line.x2) - pad))
    y0 = int(max(0, min(line.y1, line.y2) - pad))
    x1 = int(max(line.x1, line.x2) + pad)
    y1 = int(max(line.y1, line.y2) + pad)
    want = set(frames)
    tiles = []
    for idx, f in read_frames(path, max_width, frames.start, frames.stop):
        if idx not in want:
            continue
        f = f.copy()
        cv2.line(f, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, 255, 255), 1)
        crop = f[y0:y1, x0:x1]
        scale = tile_w / crop.shape[1]
        crop = cv2.resize(crop, (tile_w, int(crop.shape[0] * scale)))
        cv2.putText(crop, str(idx), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(crop)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--line", required=True, help="x1,y1,x2,y2 in the (resized) frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=150, help="frames per slit-scan image")
    ap.add_argument("--tscale", type=int, default=3, help="pixels per frame on the time axis")
    ap.add_argument("--crops", nargs=3, type=int, metavar=("FIRST", "LAST", "STEP"))
    ap.add_argument("--pad", type=int, default=60)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile-w", type=int, default=480)
    args = ap.parse_args(argv)
    line, out = Line.parse(args.line), Path(args.out)
    info = probe(args.video)
    print(f"{args.video}: {info.width}x{info.height} (source {info.source_width}x{info.source_height})")
    if args.crops:
        a, b, s = args.crops
        p = crop_tiles(
            args.video, line, range(a, b + 1, s), out / f"crops_{a:05d}_{b:05d}.jpg",
            pad=args.pad, cols=args.cols, tile_w=args.tile_w,
        )
        print(p)
        return
    scan = slit_scan(args.video, line, args.start, args.end)
    for p in render_chunks(scan, out, args.start, args.chunk, args.tscale):
        print(p)


if __name__ == "__main__":
    main()
