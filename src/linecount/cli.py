"""Command line: count objects crossing a line in a video or a live camera.

    uv run python -m linecount input.mp4 --line x1,y1,x2,y2 --out out/
    uv run python -m linecount input.mp4 --out out/ --fast --show       # draw the line with the mouse, watch live
    uv run python -m linecount 0 --out out/ --fast --show --classes people   # webcam
    uv run python -m linecount rtsp://user:pass@192.168.1.64:554/... --out out/ --fast --motion

Line coordinates are in pixels of the frame as processed (frames wider than
``--max-width`` are downscaled first; the program prints the processed size).
Without ``--line`` a window opens to draw it with the mouse.
"""

from __future__ import annotations

import argparse
import sys

from linecount.detect import MODELS, PEOPLE, VEHICLES
from linecount.pipeline import RunConfig, run
from linecount.video import probe

PRESETS = {"vehicles": VEHICLES, "people": PEOPLE}
FAST_MODEL = "yolox-tiny"
DEFAULT_THRESHOLD = {"yolox": 0.3, "rfdetr": 0.5}


def parse_classes(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in text.split(","):
        part = part.strip()
        out.extend(PRESETS.get(part, (part,)))
    return tuple(dict.fromkeys(out))


def default_threshold(model: str) -> float:
    return DEFAULT_THRESHOLD["yolox" if model.startswith("yolox") else "rfdetr"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="linecount", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file, webcam number (0) or camera URL (rtsp://...)")
    ap.add_argument("--line", default=None, help="x1,y1,x2,y2 (omit it to draw the line with the mouse)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--classes", default="vehicles", help="'vehicles', 'people' or COCO names, comma separated")
    ap.add_argument("--model", default="nano", choices=MODELS,
                    help="RF-DETR: nano/small/medium (accurate, slow on CPU); YOLOX on OpenVINO: yolox-nano/tiny/s (fast)")
    ap.add_argument("--fast", action="store_true", help=f"shortcut for --model {FAST_MODEL}")
    ap.add_argument("--device", default="CPU", help="OpenVINO device for YOLOX models: CPU or GPU (Intel graphics)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="detector confidence threshold (default: 0.5 for RF-DETR, 0.3 for YOLOX)")
    ap.add_argument("--roi", default=None, help="x1,y1,x2,y2: run the detector only on this crop")
    ap.add_argument("--anchor", default=None, choices=["bottom_center", "center"],
                    help="point of the box that must cross (default: bottom_center for vehicles, center for people)")
    ap.add_argument("--margin", type=float, default=2.0, help="dead zone around the line, in pixels")
    ap.add_argument("--stride", type=int, default=1, help="process every n-th frame (files only)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None, help="override a wrong frame rate in the file")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    ap.add_argument("--show", action="store_true", help="show the annotated frames in a window while processing")
    ap.add_argument("--motion", action="store_true", help="skip the detector while nothing moves (saves CPU on quiet cameras)")
    ap.add_argument("--realtime", action="store_true",
                    help="play a video file at its real speed, dropping the frames there is no time for, like a camera")
    ap.add_argument("--snapshots", action="store_true", help="save a photo of every crossing (always on for cameras)")
    ap.add_argument("--lang", default="en", choices=["en", "es"], help="language of the on-screen labels")
    ap.add_argument("--title", default=None, help="title shown above the counters")
    args = ap.parse_args(argv)

    model = FAST_MODEL if args.fast else args.model
    classes = parse_classes(args.classes)
    anchor = args.anchor or ("center" if classes == PEOPLE else "bottom_center")
    info = probe(args.video, args.max_width, args.fps)
    print(f"{args.video}: processing at {info.width}x{info.height}, {info.fps:.2f} fps", file=sys.stderr)
    line, roi = args.line, args.roi
    if line is None:
        from linecount.pick import first_frame, pick

        line, picked_roi = pick(first_frame(args.video, args.max_width, args.start))
        if line is None:
            print("no line drawn, nothing to do", file=sys.stderr)
            return 1
        roi = roi or picked_roi
        print(f"using --line {line}" + (f" --roi {roi}" if roi else ""), file=sys.stderr)
    cfg = RunConfig(
        line=line, model=model, device=args.device,
        threshold=args.threshold if args.threshold is not None else default_threshold(model),
        classes=classes, roi=roi, anchor=anchor, margin=args.margin, stride=args.stride, start=args.start,
        end=args.end, fps=args.fps, max_width=args.max_width, write_video=not args.no_video, title=args.title,
        show=args.show, motion=args.motion, lang=args.lang, realtime=args.realtime, snapshots=args.snapshots,
    )
    res = run(args.video, args.out, cfg)
    for name, v in res.counts.items():
        print(f"{name:<11} in {v['in']:>4}   out {v['out']:>4}")
    skipped = f", model ran on {res.detector_frames}" if args.motion else ""
    print(f"total {len(res.events)} crossings in {res.frames} frames{skipped}, {res.fps:.1f} fps -> {args.out}")
    return 0
