# SUAS 2026 — Target Localization Pipeline

Converts a flight's imagery into **one ground coordinate per target** (tent,
mannequin) for payload delivery.

The detector alone is not reliable enough per-frame — mannequin mAP@50-95 sits
near 0.30 across four architectures we tested, with false positives on bare
ground. This pipeline solves that geometrically instead: since the rules place
**exactly one** tent and one mannequin in the Search Boundary and score a drop
within **50 ft (15.24 m)**, the correct unit of decision is the *flight*, not the
frame. Detections from every frame are projected to ground coordinates and
clustered — real targets converge on one point, false positives scatter and are
rejected as noise.

**Validated result:** tent **3.80 m**, mannequin **3.94 m** on a 260 × 189 m
synthetic search area (180-frame survey) — both well inside the 50 ft radius.

---

## Install

```bash
pip install -r requirements.txt
```

Needs Python 3.9+, PyTorch, Ultralytics, scikit-learn, Pillow, numpy.
GPU optional (auto-detected; falls back to CPU).

---

## Quick start

**1. Check the flight metadata is usable** (do this before anything else):

```bash
python src/inspect_dataset.py /path/to/flight_images
```

Reports GPS, per-frame heading, altitude, camera intrinsics, and prints a
verdict. If it says heading is missing, **stop** — see *Data requirements* below.

**2. Localize:**

```bash
python src/cluster_real_yaw.py \
    --images_dir /path/to/flight_images \
    --model models/yolo11m_best.pt \
    --agl_m 50 --hfov_deg 80 \
    --conf 0.15 --eps 4.0 --min_samples 2 \
    --classifier models/mobilenet_finetuned.pth \
    --out results.json
```

Output:

```
  tent       local (  +12.2,  -18.4) m   GPS 12.9924125, 80.2360947   [14 pts]
  mannequin  local (  -38.6,   -3.0) m   GPS 12.9918441, 80.2367210   [7 pts]

  tent<->mannequin separation: 27.3 m
```

Coordinates are given in local ENU metres and as absolute GPS. Feed the GPS to
the drop planner.

**3. Inspect what happened** — open `tools/viewer.html` in a browser and load
`results.json`. Left pane shows detections per frame; right pane shows ground
coordinates accumulating and clusters forming, with the 50 ft radius drawn.

---

## Data requirements

These are hard requirements derived from a real flight that **failed** without
them. A drone flying a circuit continuously changes heading; if heading is
unknown, each detection's ground offset is rotated by the unmodelled yaw and a
*fixed* target smears across the flight path instead of converging.

| Requirement | Why |
|---|---|
| GPS per image (EXIF) | Camera position |
| **Per-frame heading** (`GPSImgDirection`, true north) | Without it, targets do not converge — this is the known failure mode |
| Height above **ground** (AGL), not GPS sea-level altitude | Sets projection scale; 20% AGL error = 20% position error |
| Camera locked nadir (or per-frame pitch/roll logged) | Determines whether the nadir assumption holds |
| Rectilinear lens mode, digital zoom off | Pinhole model degrades at frame edges under barrel distortion |
| Survey overlap giving **≥ 8 observations per target** | Fewer and no cluster forms (measured: a 48-frame pass over the same area produced none; 180 frames localized both) |
| Ground sample distance ≤ ~2 cm/px on target | At 6.6 cm/px a mannequin is ~27 px and detection fails |

`inspect_dataset.py` checks the first four automatically.

---

## Parameters

| Flag | Default | Notes |
|---|---|---|
| `--agl_m` | required | Height above ground, metres |
| `--hfov_deg` | 80 | Camera horizontal FOV. Focal length is derived from this |
| `--conf` | 0.15 | Detector threshold — **deliberately low**. Clustering rejects false positives, so the detector should maximise true-positive yield. Do not raise this to "clean up" detections |
| `--eps` | 4.0 | DBSCAN radius in metres |
| `--min_samples` | 2 | Minimum detections per cluster. Auto-relaxes if nothing clusters |
| `--rank` | `confsum` | **Leave this alone.** See below |
| `--classifier` | off | Optional MobileNet gate. Effective at matched pixel scale; rejects valid targets when they are much smaller than its training crops |

### Why `--rank confsum` matters

Clusters are ranked by **summed detection confidence**, not by point count. This
is not cosmetic. On an identical flight:

| Ranking | Mannequin error |
|---|---|
| `count` (densest cluster) | **73.60 m** — miss |
| `confsum` (default) | **0.68 m** |

Density ranking selected a chance concentration of 41 false positives over the
12 correct detections. Confidence weighting fixes it and does so at every
clustering radius tested. `count` is retained only for reproducing that
comparison.

---

## Integration

To call from flight software rather than the CLI:

```python
from src.cluster_real_yaw import run
import argparse

args = argparse.Namespace(
    images_dir="captures/", model="models/yolo11m_best.pt",
    agl_m=50.0, hfov_deg=80.0, conf=0.15, eps=4.0, min_samples=2,
    rank="confsum", classifier="models/mobilenet_finetuned.pth",
    clf_conf=0.5, out="results.json",
)
run(args)
```

Then read `results.json`:

```python
import json
r = json.load(open("results.json"))
tent_gps = r["predicted_gps"]["tent"]        # {"lat": ..., "lon": ...}
mann_gps = r["predicted_gps"]["mannequin"]
```

A class missing from `predicted_gps` means no cluster formed — treat as "target
not located", do not drop.

### `results.json` schema (relevant keys)

```
predictions      {class: [east_m, north_m]}     local ENU metres
predicted_gps    {class: {lat, lon}}            absolute
origin_latlon    [lat0, lon0]                   local frame origin
clusters[class]  points, labels, centroids, scores, point_meta
per_frame[]      image, gps, heading_deg, cam_xyz_m, detections[]
```

---

## Jetson / deployment notes

- **TensorRT engines are GPU-specific.** An engine built on a laptop will not
  load on the Orin Nano. Export on the target device.
- **Export accuracy vs. latency separately.** Accuracy degradation from
  quantization can be measured anywhere (ONNX/OpenVINO on any machine) and the
  conclusion transfers; latency must be measured on the Jetson.
- `tools/benchmark_quantization.py` runs the full pipeline once per model
  variant against the same imagery and reports localization error, detection
  yield, confidence shift, and ms/frame.
- **Watch the ranking margin**, which the benchmark reports. Because ranking is
  confidence-weighted, INT8 quantization can compress the confidence
  distribution and erode the gap between the true cluster and the runner-up.
  A passing error with a <20% margin is fragile. Standard mAP will not show
  this.
- If INT8 degrades the mannequin, fall back to FP16 — most of the speedup, far
  less accuracy risk.

---

## Files

```
src/cluster_pipeline.py    clustering core, back-projection, classifier gate
src/cluster_real_yaw.py    real-flight entry point (EXIF GPS + heading)  <- run this
src/inspect_dataset.py     metadata validator — run before any new flight
tools/viewer.html          replay UI, loads results.json, no GPU needed
tools/benchmark_quantization.py   quantized-variant comparison
models/                    detector + verification classifier weights
```

`cluster_real_yaw.py` imports from `cluster_pipeline.py` as a sibling — keep
them in the same directory.

---

## Known limitations

- Assumes a **flat ground plane** at the targets' elevation. Significant terrain
  relief introduces proportional position error.
- Assumes a **pinhole camera**. Strong barrel distortion degrades accuracy for
  detections near frame edges.
- The verification classifier is **scale-sensitive** — it rejects valid targets
  rendered much smaller than its training crops. Disable it rather than fight it
  if target pixel size is low; the geometric false-positive rejection works
  without it.
