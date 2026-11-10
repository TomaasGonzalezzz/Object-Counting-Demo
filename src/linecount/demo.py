"""One command to show the counter working on the three public clips, live in a window.

    uv run python -m linecount.demo                     # all three clips, fast model, Spanish labels
    uv run python -m linecount.demo --clip a40          # one clip
    uv run python -m linecount.demo --accurate          # RF-DETR instead of YOLOX (slow on a laptop)
    uv run python -m linecount.demo --no-show           # only render the annotated videos
    uv run python -m linecount.demo --clip a40 --realtime   # play at real speed, as a camera would

Clips are downloaded from Wikimedia Commons the first time. Each clip uses the
same line and settings as the evaluation (``evaluation/configs.json``: ``fast``,
or ``final`` with ``--accurate``). Annotated videos and CSVs go to ``out/demo/``.
``q`` or ``Esc`` in the window skips to the next clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from linecount.evaluate import ROOT, clip_run_config, load_json

TITLES = {
    "en": {"a40": "Motorway A40, Germany", "roundabout": "Roundabout from a drone, USA",
           "pedestrian_area": "Pedestrian street, Munich"},
    "es": {"a40": "Autopista A40, Alemania", "roundabout": "Rotonda vista desde un dron, EE.UU.",
           "pedestrian_area": "Calle peatonal, Munich"},
}


def default_data_dir() -> Path:
    shared = ROOT.parent / "external-data" / "counting" / "raw"
    return shared if shared.exists() else ROOT / "data"


def main(argv=None) -> int:
    clips = load_json("clips.json")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="all", choices=["all", *clips])
    ap.add_argument("--accurate", action="store_true", help="RF-DETR Nano (the 'final' configuration) instead of YOLOX-tiny")
    ap.add_argument("--device", default="CPU", help="OpenVINO device for the fast model: CPU or GPU (Intel graphics)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window, only write the videos")
    ap.add_argument("--realtime", action="store_true",
                    help="play each clip at its real speed, as a camera would (only the frames there is time for are analysed)")
    ap.add_argument("--lang", default="es", choices=["es", "en"])
    ap.add_argument("--data", type=Path, default=None, help="folder with the clips (downloaded there if missing)")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "demo")
    args = ap.parse_args(argv)

    from linecount.data import download
    from linecount.pipeline import run

    data = args.data or default_data_dir()
    names = list(clips) if args.clip == "all" else [args.clip]
    if any(not (data / clips[n]["file"]).exists() for n in names):
        download(data)
    config = "final" if args.accurate else "fast"
    for name in names:
        cfg = clip_run_config(config, name, title=TITLES[args.lang][name], show=not args.no_show, lang=args.lang,
                              realtime=args.realtime, snapshots=True)
        if not args.accurate:
            cfg.device = args.device
        start, end = clips[name]["window"]
        cfg.start, cfg.end = start, end
        out = args.out / f"{name}_{config}{'_realtime' if args.realtime else ''}"
        print(f"{name}: {cfg.model}, line {cfg.line}" + (f", crop {cfg.roi}" if cfg.roi else ""), flush=True)
        res = run(data / clips[name]["file"], out, cfg)
        counted = ", ".join(f"{c} {v['in'] + v['out']}" for c, v in res.counts.items()) or "nothing"
        print(f"  counted: {counted}  |  {res.fps:.1f} frames/s  |  {out / 'annotated.mp4'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
