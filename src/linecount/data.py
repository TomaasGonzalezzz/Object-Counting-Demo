"""Download the evaluation clips from Wikimedia Commons.

The clips are not stored in this repository. This downloads the original files
listed in ``evaluation/clips.json`` into a folder of your choice (each clip in
its own subfolder), skipping files that are already there.

    uv run python -m linecount.data --out ../external-data/counting/raw
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from linecount.evaluate import load_json

USER_AGENT = "linecount-demo/0.1 (evaluation clip download)"


def download(out: Path) -> None:
    for name, clip in load_json("clips.json").items():
        dest = out / clip["file"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{name}: already at {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {clip['url']}")
        req = urllib.request.Request(clip["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB, {clip['license']}, {clip['author']}")
        time.sleep(2)  # be polite to upload.wikimedia.org


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    download(Path(ap.parse_args(argv).out))


if __name__ == "__main__":
    main()
