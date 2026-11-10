# Data: clips, licenses and model weights

No video is stored in this repository. The three evaluation clips are downloaded from Wikimedia Commons with

```sh
uv run python -m linecount.data --out ../external-data/counting/raw
```

and the RF-DETR weights are downloaded by `rfdetr` on first use into the folder set by `RF_HOME` (I used `../external-data/counting/weights`).

## Clips

| Clip | What it shows | Length used | Source | License | Author |
| --- | --- | --- | --- | --- | --- |
| `a40` | Motorway A40 (Ruhr area, Germany) seen from a bridge, both directions | 24.5 s, 735 frames, 1280x720, 30 fps | [Commons](https://commons.wikimedia.org/wiki/File:Motorway_A40_-_on_bridge_above_the_traffic.webm) (originally [soundsofchanges.eu](http://www.soundsofchanges.eu/)) | CC BY 3.0 | Kathinka Engels |
| `roundabout` | Roundabout on US 14/61 in La Crosse, Wisconsin, seen from a drone; the counting line is on the west approach | 36.4 s, 872 frames, 1920x1080 downscaled to 1280x720, 23.98 fps | [Commons](https://commons.wikimedia.org/wiki/File:Roundabout_14_61.webm) | CC0 1.0 | Wikideas1 |
| `pedestrian_area` | Pedestrian street (Neuhauser Straße, Munich), low static camera, people very close to the lens | 15 s, 375 frames, 1920x1080 downscaled to 1280x720, 25 fps | [Commons](https://commons.wikimedia.org/wiki/File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm) (originally the [Xiph.org test media collection](https://media.xiph.org/video/derf/)) | CC0 1.0 (as stated on Commons) | Taurus Media Technik |

Notes:

- The whole file is used for each clip; nothing was trimmed. Frames wider than 1280 px are downscaled on the fly while reading.
- The `pedestrian_area` WebM reports 1000 fps in its container. The real rate is 25 fps, set in `evaluation/clips.json`.
- Clips considered and dropped: *Road traffic on Stritarjeva street* (cuts between camera positions), *Avtocesta* (night, camera pans), the Singapore and Toronto scramble crossings (hand-held camera).
- Images and the GIF in `figures/` are frames of these clips with the program's overlay drawn on top. For `a40` that is a derivative of a CC BY 3.0 work: credit "Motorway A40 - on bridge above the traffic" by Kathinka Engels, CC BY 3.0, modified (overlay added). The other two are CC0 and need no credit, but it is given anyway.

## Ground truth

`evaluation/ground_truth/*.csv` are my own annotations (frame numbers, directions and classes of each crossing), made by hand as described in [EVALUATION.md](EVALUATION.md). They contain no image data.

## Software and model licenses

| Component | Used for | License |
| --- | --- | --- |
| [rfdetr](https://github.com/roboflow/rf-detr) 1.12 and the RF-DETR Nano COCO weights | detection | Apache-2.0 |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) nano / tiny / s, ONNX files from release 0.1.1rc0 (COCO weights) | fast detection (`--fast`, `--model yolox-*`), downloaded on first use | Apache-2.0 |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) 2026.4 | runs YOLOX on the CPU or Intel graphics | Apache-2.0 |
| [trackers](https://github.com/roboflow/trackers) 2.7 (ByteTrack) | tracking | Apache-2.0 |
| [supervision](https://github.com/roboflow/supervision) 0.31 | detection containers, NMS | MIT |
| PyTorch, torchvision | rfdetr runtime | BSD-3-Clause |
| transformers | rfdetr dependency | Apache-2.0 |
| OpenCV (`opencv-python`) | video I/O, drawing | Apache-2.0 (the wheels bundle FFmpeg under LGPL) |
| NumPy, SciPy | geometry, event matching | BSD-3-Clause |
| Pillow | GIF writing | MIT-CMU (HPND) |
| Matplotlib | evaluation figure | Matplotlib license (PSF-style, BSD-compatible) |
| pytest | tests | MIT |
| PyAV (optional `web` group only) | encoding the H.264 web clip | BSD-3-Clause; its wheels bundle an FFmpeg build that includes x264/x265 (GPL). The clip in `figures/web/` was encoded with the Windows Media Foundation encoder (`h264_mf`), not x264 |

Nothing the counting code imports is AGPL or GPL. Ultralytics YOLO models were deliberately not used (AGPL-3.0). The larger RF-DETR XL / 2XL models ship under a different, non-Apache license in `rfdetr[plus]` and were not installed.
