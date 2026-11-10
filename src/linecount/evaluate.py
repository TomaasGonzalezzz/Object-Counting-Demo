"""Evaluation against the manual ground truth.

Ground truth lives in ``evaluation/ground_truth/<clip>.csv``: one row per object
that crosses the line, with frame, direction, position along the line and
class (see docs/EVALUATION.md for how it was made). Clips and their lines are in
``evaluation/clips.json`` and detector/tracker settings in
``evaluation/configs.json``.

Predicted crossings are matched to true crossings one to one (Hungarian
assignment) when they have the same direction, happen within
``tolerance_frames`` of each other and lie within ``tolerance_px`` along the
line. Then, inside the clip's evaluation window:

* matched pairs are correct crossings;
* unmatched true crossings are **missed**;
* unmatched predictions are **double counts** when a matched prediction already
  covers a true crossing within the tolerance (the same object counted twice,
  usually after an ID switch), otherwise **false crossings**.

Rows of class ``ignore`` in the ground truth mark objects too small to identify:
a prediction matched to one is neither rewarded nor penalised.

    uv run python -m linecount.evaluate run --config tuned --data <clips folder>
    uv run python -m linecount.evaluate summary --config tuned
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "evaluation"
RESULTS_DIR = ROOT / "results"


def load_gt(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame"] = int(r["frame"])
        r["position"] = float(r["position"])
    return rows


def load_events(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame"] = int(r["frame"])
        r["position"] = float(r["position"])
        r["track_id"] = int(r["track_id"])
    return rows


def match_events(gt: list[dict], pred: list[dict], tol_frames: float, tol_px: float) -> list[tuple[int, int]]:
    """Return matched (gt_index, pred_index) pairs."""
    if not gt or not pred:
        return []
    big = 1e6
    cost = np.full((len(gt), len(pred)), big)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            dt = abs(g["frame"] - p["frame"])
            dp = abs(g["position"] - p["position"])
            if g["direction"] == p["direction"] and dt <= tol_frames and dp <= tol_px:
                cost[i, j] = dt / tol_frames + dp / tol_px
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < big]


def score(gt: list[dict], pred: list[dict], window: tuple[int, int], tol_frames: float, tol_px: float) -> dict:
    """Count-level and event-level metrics for the events inside ``window`` = [start, end)."""
    lo, hi = window
    pairs = match_events(gt, pred, tol_frames, tol_px)
    gt_to_pred = {i: j for i, j in pairs}
    pred_to_gt = {j: i for i, j in pairs}

    def inside(frame):
        return lo <= frame < hi

    rows = []
    correct = missed = double = false = class_ok = 0
    for i, g in enumerate(gt):
        if g["class"] == "ignore" or not inside(g["frame"]):
            continue
        if i in gt_to_pred:
            p = pred[gt_to_pred[i]]
            correct += 1
            same = p["class"] == g["class"]
            class_ok += same
            rows.append({"status": "correct", "gt_event": g["event"], "gt_frame": g["frame"], "pred_frame": p["frame"],
                         "track_id": p["track_id"], "direction": g["direction"], "gt_class": g["class"],
                         "pred_class": p["class"], "class_ok": same})
        else:
            missed += 1
            rows.append({"status": "missed", "gt_event": g["event"], "gt_frame": g["frame"], "pred_frame": "",
                         "track_id": "", "direction": g["direction"], "gt_class": g["class"], "pred_class": "",
                         "class_ok": ""})
    for j, p in enumerate(pred):
        if j in pred_to_gt:
            continue  # matched: already reported (or matched to an ignore / out-of-window object)
        if not inside(p["frame"]):
            continue
        near_matched = any(
            g["direction"] == p["direction"]
            and abs(g["frame"] - p["frame"]) <= tol_frames
            and abs(g["position"] - p["position"]) <= tol_px
            and i in gt_to_pred
            for i, g in enumerate(gt)
        )
        status = "double_count" if near_matched else "false_crossing"
        double += near_matched
        false += not near_matched
        rows.append({"status": status, "gt_event": "", "gt_frame": "", "pred_frame": p["frame"],
                     "track_id": p["track_id"], "direction": p["direction"], "gt_class": "",
                     "pred_class": p["class"], "class_ok": ""})

    # A matched pair is attributed to the window by its true crossing frame, an unmatched prediction by its own
    # frame, so a crossing near the window edge is never counted on one side only.
    true_count = sum(1 for g in gt if g["class"] != "ignore" and inside(g["frame"]))
    pred_count = correct + double + false

    per_class_true: dict[str, int] = {}
    per_class_pred: dict[str, int] = {}
    for g in gt:
        if g["class"] != "ignore" and inside(g["frame"]):
            per_class_true[g["class"]] = per_class_true.get(g["class"], 0) + 1
    for r in rows:
        if r["pred_class"]:
            per_class_pred[r["pred_class"]] = per_class_pred.get(r["pred_class"], 0) + 1

    err = pred_count - true_count
    return {
        "window": [lo, hi],
        "true_count": true_count,
        "predicted_count": pred_count,
        "error": err,
        "abs_error": abs(err),
        "pct_error": round(100 * err / true_count, 1) if true_count else None,
        "correct": correct,
        "missed": missed,
        "double_counts": double,
        "false_crossings": false,
        "recall": round(correct / true_count, 3) if true_count else None,
        "precision": round(correct / (correct + double + false), 3) if (correct + double + false) else None,
        "class_accuracy": round(class_ok / correct, 3) if correct else None,
        "per_class_true": dict(sorted(per_class_true.items())),
        "per_class_predicted": dict(sorted(per_class_pred.items())),
        "events": rows,
    }


# ----------------------------------------------------------------------------- running


def load_json(name: str) -> dict:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def clip_run_config(config_name: str, clip_name: str, **overrides):
    """The ``RunConfig`` a named configuration uses on a clip (clip line and classes + config settings)."""
    from linecount.cli import parse_classes
    from linecount.pipeline import RunConfig

    clip = load_json("clips.json")[clip_name]
    base = load_json("configs.json")[config_name]
    settings = {**base.get("default", {}), **base.get(clip["group"], {}), **base.get(clip_name, {})}
    roi = settings.pop("roi", None)
    if roi == "auto":
        roi = clip.get("roi")
    settings = {"title": None, **settings, **overrides}
    return RunConfig(line=clip["line"], classes=parse_classes(clip["classes"]), anchor=clip["anchor"],
                     fps=clip.get("fps"), roi=roi, **settings)


def run_config(config_name: str, clip_names: list[str], data_dir: Path, runs_dir: Path, write_video: bool,
               show: set[str] | None = None) -> None:
    from linecount.pipeline import run

    clips = load_json("clips.json")
    for name in clip_names:
        clip = clips[name]
        cfg = clip_run_config(config_name, name, write_video=write_video)
        out = runs_dir / config_name / name
        print(f"[{config_name}] {name}: {cfg}", flush=True)
        res = run(data_dir / clip["file"], out, cfg)
        print(f"[{config_name}] {name}: {len(res.events)} crossings, {res.fps:.2f} fps", flush=True)
        dest = RESULTS_DIR / config_name
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ("events.csv", "counts.csv", "run.json"):
            (dest / f"{name}_{fname}").write_bytes((out / fname).read_bytes())
    score_config(config_name, show)


def score_config(config_name: str, show: set[str] | None = None) -> dict:
    clips = load_json("clips.json")
    dest = RESULTS_DIR / config_name
    metrics: dict = {}
    for name, clip in clips.items():
        events_path = dest / f"{name}_events.csv"
        if not events_path.exists():
            continue
        gt = load_gt(EVAL_DIR / "ground_truth" / f"{name}.csv")
        pred = load_events(events_path)
        run_info = json.loads((dest / f"{name}_run.json").read_text(encoding="utf-8"))
        splits = clip.get("splits", {"all": clip["window"]})
        clip_metrics = {"speed": {k: run_info.get(k) for k in ("frames_processed", "wall_seconds", "pipeline_fps",
                                                               "video_frames_per_second", "detector_ms_per_frame")}}
        for split, window in {"all": clip["window"], **splits}.items():
            m = score(gt, pred, tuple(window), clip["tolerance_frames"], clip["tolerance_px"])
            rows = m.pop("events")
            clip_metrics[split] = m
            with open(dest / f"{name}_{split}_matches.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["status", "gt_event", "gt_frame", "pred_frame", "track_id",
                                                  "direction", "gt_class", "pred_class", "class_ok"])
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: (r["gt_frame"] or r["pred_frame"])))
        metrics[name] = clip_metrics
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _print_table(config_name, metrics, clips, show)
    return metrics


def _print_table(config_name: str, metrics: dict, clips: dict, show: set[str] | None = None) -> None:
    print(f"\n{config_name}")
    print(f"{'clip':<16}{'split':<7}{'true':>5}{'pred':>6}{'err%':>7}{'ok':>4}{'miss':>5}{'dbl':>5}{'false':>6}{'cls':>6}{'fps':>6}")
    for name, m in metrics.items():
        for split, s in m.items():
            if split == "speed" or (split == "all" and len(m) > 2 and "splits" in clips[name]):
                continue
            if show and split not in show:
                continue
            pct = "" if s["pct_error"] is None else f"{s['pct_error']:+.0f}"
            cls = "" if s["class_accuracy"] is None else f"{s['class_accuracy']:.2f}"
            print(f"{name:<16}{split:<7}{s['true_count']:>5}{s['predicted_count']:>6}{pct:>7}{s['correct']:>4}"
                  f"{s['missed']:>5}{s['double_counts']:>5}{s['false_crossings']:>6}{cls:>6}{m['speed']['pipeline_fps']:>6}")


def _summary_row(config_name: str, name: str, clips: dict) -> dict:
    m = json.loads((RESULTS_DIR / config_name / "metrics.json").read_text(encoding="utf-8"))[name]
    clip = clips[name]
    not_blind = load_json("configs.json")[config_name].get("_not_blind_on", [])
    split = "test" if "test" in m else "all"
    s = m[split]
    return {
        "config": config_name,
        "clip": name,
        "title": clip["title"],
        "objects": clip["classes"],
        "evaluated_frames": f"{s['window'][0]}-{s['window'][1]}",
        "held_out": (split == "test" or not clip.get("used_for_tuning", False)) and name not in not_blind,
        "true_count": s["true_count"],
        "predicted_count": s["predicted_count"],
        "pct_error": s["pct_error"],
        "correct": s["correct"],
        "missed": s["missed"],
        "double_counts": s["double_counts"],
        "false_crossings": s["false_crossings"],
        "class_accuracy": s["class_accuracy"],
        "per_class_true": s["per_class_true"],
        "per_class_predicted": s["per_class_predicted"],
        "frames_per_second_cpu": m["speed"]["pipeline_fps"],
        "detector_ms_per_frame": m["speed"]["detector_ms_per_frame"],
        "source_url": clip["page"],
        "license": clip["license"],
        "author": clip["author"],
    }


def write_summary(config_name: str, out: Path, also: list[str] | None = None) -> None:
    """Headline numbers for the README and the website (results/summary.json)."""
    clips = load_json("clips.json")
    metrics = json.loads((RESULTS_DIR / config_name / "metrics.json").read_text(encoding="utf-8"))
    rows = [_summary_row(config_name, name, clips) for name in metrics]
    other = []
    for item in also or []:
        cfg, name = item.split(":")
        other.append(_summary_row(cfg, name, clips))
    summary = {
        "config": config_name,
        "hardware": "Intel Core i7-10510U laptop CPU (4 cores), no GPU",
        "detector": "RF-DETR (rfdetr, Apache-2.0), COCO-pretrained, no fine-tuning",
        "tracker": "ByteTrack (Roboflow trackers, Apache-2.0)",
        "note": "Counts compared with a manual ground truth; see docs/EVALUATION.md.",
        "clips": rows,
        "comparison_runs": other,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a configuration on clips and score it")
    r.add_argument("--config", required=True)
    r.add_argument("--clips", default=None, help="comma separated (default: all)")
    r.add_argument("--data", required=True, help="folder with the downloaded clips (see linecount.data)")
    r.add_argument("--runs", default=None, help="where annotated videos go (default: <data>/../runs)")
    r.add_argument("--no-video", action="store_true")
    r.add_argument("--show", default=None, help="only print these splits, e.g. 'tune' while tuning")
    s = sub.add_parser("score", help="re-score saved events without running the model")
    s.add_argument("--config", required=True)
    s.add_argument("--show", default=None)
    m = sub.add_parser("summary", help="write results/summary.json")
    m.add_argument("--config", required=True)
    m.add_argument("--also", default=None, help="extra config:clip pairs to list as comparison runs")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        clips = args.clips.split(",") if args.clips else list(load_json("clips.json"))
        data = Path(args.data)
        runs = Path(args.runs) if args.runs else data.parent / "runs"
        show = set(args.show.split(",")) if args.show else None
        run_config(args.config, clips, data, runs, not args.no_video, show)
    elif args.cmd == "score":
        score_config(args.config, set(args.show.split(",")) if args.show else None)
    else:
        write_summary(args.config, RESULTS_DIR / "summary.json", args.also.split(",") if args.also else None)


if __name__ == "__main__":
    main()
