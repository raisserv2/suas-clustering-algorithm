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
synthetic search area (180-frame survey), rendered with SIYI A8 mini optics —
i.e. our flight camera. Both well inside the 50 ft radius.

---

## How it works

### Pipeline

```
   flight images (+ EXIF GPS, heading, AGL)
        │
        ▼
   ┌─────────────┐   per frame, in capture order
   │  1 DETECT   │   YOLO at low confidence (0.15)
   └─────────────┘   → boxes + class + confidence
        │
        ▼
   ┌─────────────┐   optional (--classifier)
   │ 2 CLF GATE  │   MobileNetV3 re-checks each crop
   └─────────────┘   → drop anything classed "background"
        │
        ▼
   ┌─────────────┐   box centre pixel → ground coordinate
   │ 3 GEOLOCATE │   using camera pose + intrinsics + AGL
   └─────────────┘   → (East, North) metres
        │
        ▼            accumulate across ALL frames
   ┌─────────────┐   DBSCAN per class
   │  4 CLUSTER  │   → clusters + noise
   └─────────────┘
        │
        ▼
   ┌─────────────┐   score = Σ member confidences
   │   5 RANK    │   → best cluster per class
   └─────────────┘
        │
        ▼
   one (lat, lon) per class  →  drop planner
```

Stages 1–3 run per frame; stages 4–5 run once at the end over everything
accumulated. The classifier gate sits **before** geolocation deliberately, so
rejected detections never enter the point cloud at all.

### 1. Detection — deliberately permissive

The detector runs at `--conf 0.15`, much lower than typical. This is
intentional and worth understanding before anyone "fixes" it: false-positive
rejection happens geometrically in stage 4, so the detector's job here is to
maximise *true-positive yield* feeding the cluster, not to be selective.
Raising the threshold starves the cluster of the points it needs.

### 2. Classifier gate — optional

Each detection crop is re-classified by a MobileNetV3-Large head
(tent / mannequin / background). Anything classed background, or below
`--clf_conf`, is discarded. When the classifier fires confidently it also
overrides the detector's class label.

This stage is **scale-sensitive**: it works when the target occupies a similar
pixel scale to its training crops, and rejects valid targets when they are far
smaller. At A8 ≤150 ft (87+ px) it is in range. It is optional because the
primary false-positive defence is geometric and works without it.

### 3. Geolocation — ray-cast to ground plane

Each detection's box-centre pixel is converted to a ground coordinate:

1. Build a ray in camera space from the pixel offset and the focal length
   (derived from `--hfov_deg` and image width).
2. Rotate that ray into world space using the camera's orientation — nadir
   pitch and zero roll, held fixed by the gimbal, plus the per-frame heading
   (`GPSImgDirection` for real flights).
3. Intersect it with the ground plane at target elevation.
4. Offset by the camera's own position (GPS → local ENU metres).

**Gimbal model: yaw follows the drone, pitch and roll don't.** A gimbal holds
the camera level and pointing straight down regardless of the drone body's
attitude — only yaw (heading) tracks the drone. `backproject_raycast` (used
for synthetic flights, which supply a full drone pose matrix) reflects this:
it extracts just the yaw from that pose and rebuilds a nadir, zero-roll
orientation from it, rather than trusting whatever pitch/roll the drone body
happened to have in a given frame. `backproject_yaw_nadir` (used for real
flights, which supply an explicit `GPSImgDirection` heading instead of a full
pose matrix) already worked this way. The two are numerically identical for
the same yaw — confirmed by direct comparison across a range of headings, and
confirmed that injecting an artificial drone pitch into the pose matrix no
longer perturbs `backproject_raycast`'s output at all.

Verified numerically:

| Check | Result |
|---|---|
| Forward-project a known point → pixel → back-project | ≤ 1.1 × 10⁻¹⁴ m |
| Ray-cast vs. closed-form nadir formula, nadir camera | 5.0 × 10⁻¹⁵ m |
| Fixed target from 5 different positions **and** headings | converge, spread ≤ 1.5 × 10⁻¹⁴ m |

Those are floating-point zero. **The geolocation stage is exact**, so any field
error is attributable to pose measurement or detection — never to the
projection. That separation is what made debugging tractable.

> The earlier version of this pipeline let a drone-body pitch/roll baked into
> the synthetic pose matrix rotate the ray along with yaw, and had a
> validation row ("known target under yaw 0–270°, pitch 0–25°, roll ±8°")
> exercising that. That row described a mode that no longer exists — pitch
> and roll are now fixed by the gimbal model above — so it's been removed
> rather than left describing behavior the code no longer has.

The last row is the property the whole method rests on: the same physical
target, seen from anywhere at any heading, lands on the same ground coordinate.

### 4. Clustering — DBSCAN

Ground coordinates are accumulated per class across the entire flight, then
clustered with DBSCAN (`--eps` metres, `--min_samples` points).

DBSCAN is chosen over k-means for two specific reasons:

- It does not require the number of clusters up front. We do not know how many
  distinct things the detector fired on.
- It labels low-density points as **noise**. Scattered false positives are
  therefore rejected by the algorithm's own behaviour rather than by a
  hand-tuned filter. This *is* the false-positive rejection mechanism.

A real target, detected across many frames, produces a tight knot of points at
one coordinate. A false positive on some ground feature appears in one frame and
projects somewhere unrelated in the next, so it never accumulates density.

### 5. Ranking — confidence-weighted, not densest

Each surviving cluster is scored as the **sum of its members' detection
confidences**, and the winning cluster's centroid is a confidence-weighted mean.
The highest-scoring cluster per class is emitted as that target's location.

The obvious alternative — pick the cluster with the most points — fails. On an
identical flight:

| Ranking | Mannequin error |
|---|---|
| `count` (densest cluster) | **73.60 m** — miss |
| `confsum` (default) | **0.68 m** |

Diagnosis of that failure: of 53 mannequin points, 12 lay within 50 ft of truth
and the closest was 0.6 m — the detector had found it. But among the 41 false
positives was a chance concentration containing *more points* than the true
cluster. The detection was right; the ranking picked the wrong cluster.

Real detections and ground false positives differ systematically in confidence
even when they do not differ in count, so scoring by summed confidence lets
quality outvote quantity. Sweeps confirm it selects correctly at **every**
`--eps` tested (1.0, 1.5, 2.0 m), whereas density ranking only recovers the
right answer at the tightest setting — so this is robust, not tuned.

### Robustness: sparse-target relaxation

A target detected in only 2–3 frames can form a perfectly tight, accurate
cluster that DBSCAN discards for falling below `min_samples` — a correct answer
thrown away on a technicality.

If no cluster forms for a class while points exist, clustering is retried with
`min_samples = 1`. The winner must then pass a viability test (≥ 2 member points,
or a single detection above a confidence floor). This recovered a three-point
tent cluster to 0.42 m in testing, without admitting isolated low-confidence
false positives.

### Two entry points

| Script | Use for | Pose source |
|---|---|---|
| `src/cluster_real_yaw.py` | **real flights** | EXIF GPS + `GPSImgDirection` heading + `--agl_m` |
| `src/cluster_pipeline.py synthetic` | pre-rendered synthetic flights | `flight_log.jsonl` camera poses; reports error against known ground truth |

Both share the same stages 1–5; only the pose source differs. The synthetic mode
is what produced the 3.80 / 3.94 m validation figures, since it has ground truth
to measure against.

---

## Camera: SIYI A8 mini

**The pipeline is camera-agnostic.** Intrinsics are passed at runtime, not
hardcoded, which is why the same code has already run against three different
cameras (A8 mini, GoPro HERO13, a stock 720p unit). The A8 mini is our flight
camera and the configuration everything is validated against.

### Where the A8 mini is assumed, and where it isn't

| Component | A8-specific? | Notes |
|---|---|---|
| `src/cluster_pipeline.py` | **No** | Reads intrinsics from the flight log or CLI |
| `src/cluster_real_yaw.py` | **No** | Takes `--hfov_deg`; pass any camera's FOV |
| `src/inspect_dataset.py` | **No** | Reads whatever is in the EXIF |
| `tools/viewer.html` | **No** | Renders whatever is in `results.json` |
| Synthetic flight renderer *(separate repo)* | **Yes** | `CAMERA_CONFIG` hardcodes A8 sensor geometry; this is what the 3.80/3.94 m result was rendered with |

So: nothing in **this** repo is tied to the A8. The dependency is only that our
validation imagery was generated with A8 optics, which means the reported
accuracy figures apply directly to A8 flights and would need re-measuring for a
different camera.

### Verified A8 mini parameters

Derived from the sensor geometry (7.60 × 5.70 mm sensor, 4.51 mm focal length,
3840 × 2160):

| Quantity | Value |
|---|---|
| Horizontal FOV | **80.2°** |
| Vertical FOV | 64.6° |
| Focal length | 2278.7 px |
| Resolution | 3840 × 2160 |

Pass `--hfov_deg 80.2`. The pipeline derives focal length from FOV and image
width, so that single flag is all the camera configuration required.

> The A8 mini's "6× zoom" is **digital only** — a pixel crop with no additional
> resolving power. Do not treat it as optical zoom, and do not enable it during
> capture: cropping reduces the footprint without adding target detail, and
> changes the effective FOV so `--hfov_deg 80.2` would no longer be correct.

### Footprint and target size by altitude

| Altitude | Ground footprint | GSD | Mannequin | Tent |
|---|---|---|---|---|
| 127 ft (38.7 m) | 65.2 × 48.9 m | 1.70 cm/px | 103 px | 147 px |
| **150 ft (45.7 m)** | 77.0 × 57.8 m | 2.01 cm/px | **87 px** | **125 px** |
| 200 ft (61.0 m) | 102.7 × 77.0 m | 2.68 cm/px | 65 px | 93 px |
| 250 ft (76.2 m) | 128.4 × 96.3 m | 3.34 cm/px | 52 px | 75 px |

The 127 ft row is the configuration the 3.80/3.94 m validation was flown at.
At the 150 ft competition floor the mannequin is 87 px — comfortably workable,
and far above the ~27 px at which detection failed on an earlier real flight.

**Altitude is the main accuracy lever.** Detection degrades with target pixel
size, so fly as close to the 150 ft floor as the rules and flight plan allow.
Above ~200 ft the mannequin drops below 65 px and detection reliability falls
off; treat that as a soft ceiling.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+, PyTorch, Ultralytics, scikit-learn, Pillow, numpy.
GPU optional (auto-detected, falls back to CPU).

---

## Quick start

**1. Check the flight metadata is usable** — do this before anything else:

```bash
python src/inspect_dataset.py /path/to/flight_images
```

Reports GPS, per-frame heading, altitude, intrinsics, and prints a verdict.
If it says heading is missing, **stop** — see *Data requirements*.

**2. Localize** (A8 mini at 150 ft):

```bash
python src/cluster_real_yaw.py \
    --images_dir /path/to/flight_images \
    --model models/yolo11m_best.pt \
    --agl_m 45.7 --hfov_deg 80.2 \
    --conf 0.15 --eps 4.0 --min_samples 2 \
    --classifier models/mobilenet_finetuned.pth \
    --out results.json
```

`--agl_m` is height above **ground** in metres (150 ft = 45.7 m), not GPS
sea-level altitude.

Output:

```
  tent       local (  +12.2,  -18.4) m   GPS 12.9924125, 80.2360947   [14 pts]
  mannequin  local (  -38.6,   -3.0) m   GPS 12.9918441, 80.2367210   [7 pts]

  tent<->mannequin separation: 27.3 m
```

Feed `predicted_gps` to the drop planner.

**3. Inspect** — open `tools/viewer.html` and load `results.json`. Left pane:
detections per frame. Right pane: ground coordinates accumulating into clusters,
with the 50 ft radius drawn.

---

## Data requirements

Derived from a real flight that **failed** without them. A drone flying a
circuit continuously changes heading; if heading is unknown, each detection's
ground offset is rotated by the unmodelled yaw and a *fixed* target smears
across the flight path instead of converging.

| Requirement | Why |
|---|---|
| GPS per image (EXIF) | Camera position |
| **Per-frame heading** (`GPSImgDirection`, true north) | Without it targets do not converge — the known failure mode |
| Height above **ground** (AGL) | Sets projection scale; 20% AGL error = 20% position error |
| Camera gimbal-locked nadir, level roll | The pipeline assumes pitch/roll are held constant by the gimbal and only yaw tracks the drone — it does not model a drone body that pitches/rolls the camera with it |
| Digital zoom **off**, no crop modes | Changes effective FOV and invalidates `--hfov_deg` |
| Survey overlap giving **≥ 8 observations per target** | Measured: a 48-frame pass produced no cluster; 180 frames localized both |
| GSD ≤ ~2 cm/px on target | A8 at ≤150 ft satisfies this; at 6.6 cm/px detection failed |

`inspect_dataset.py` checks the first four automatically.

---

## Parameters

| Flag | A8 value | Notes |
|---|---|---|
| `--agl_m` | 45.7 (at 150 ft) | Height above ground, metres |
| `--hfov_deg` | **80.2** | A8 mini, digital zoom off |
| `--conf` | 0.15 | **Deliberately low.** Clustering rejects false positives, so the detector should maximise true-positive yield. Do not raise this to "clean up" detections |
| `--eps` | 4.0 | DBSCAN radius, metres |
| `--min_samples` | 2 | Minimum per cluster; auto-relaxes if nothing clusters |
| `--rank` | `confsum` | **Leave this alone** — see below |
| `--classifier` | optional | MobileNet gate. Works at matched pixel scale; rejects valid targets when they are far smaller than its training crops. At A8 ≤150 ft (87+ px) it is in range and worth enabling |

### Why `--rank confsum` matters

Clusters are ranked by **summed detection confidence**, not point count. On an
identical flight:

| Ranking | Mannequin error |
|---|---|
| `count` (densest cluster) | **73.60 m** — miss |
| `confsum` (default) | **0.68 m** |

Density ranking selected a chance concentration of 41 false positives over the
12 correct detections. Confidence weighting fixes it, at every clustering radius
tested. `count` is retained only to reproduce that comparison.

---

## Integration

```python
from src.cluster_real_yaw import run
import argparse

args = argparse.Namespace(
    images_dir="captures/", model="models/yolo11m_best.pt",
    agl_m=45.7, hfov_deg=80.2,          # A8 mini @ 150 ft
    conf=0.15, eps=4.0, min_samples=2, rank="confsum",
    classifier="models/mobilenet_finetuned.pth", clf_conf=0.5,
    out="results.json",
)
run(args)
```

```python
import json
r = json.load(open("results.json"))
tent_gps = r["predicted_gps"].get("tent")        # {"lat":..., "lon":...} or None
mann_gps = r["predicted_gps"].get("mannequin")
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
  load on the Orin Nano — export on the target device.
- **Measure accuracy and latency separately.** Quantization accuracy loss can be
  measured anywhere (ONNX/OpenVINO) and the conclusion transfers; latency must
  be measured on the Jetson.
- `tools/benchmark_quantization.py` runs the full pipeline once per model
  variant against identical imagery, reporting localization error, detection
  yield, confidence shift, and ms/frame.
- **Watch the ranking margin** the benchmark reports. Because ranking is
  confidence-weighted, INT8 can compress the confidence distribution and erode
  the gap between the true cluster and the runner-up. A passing error with a
  <20% margin is fragile — standard mAP will not reveal this.
- If INT8 degrades the mannequin, fall back to FP16.

---

## Files

```
src/cluster_pipeline.py           clustering core, back-projection, classifier gate
src/cluster_real_yaw.py           real-flight entry point (EXIF GPS + heading)  <- run this
src/inspect_dataset.py            metadata validator — run before any new flight
tools/viewer.html                 replay UI, loads results.json, no GPU needed
tools/benchmark_quantization.py   quantized-variant comparison
models/                           detector + verification classifier weights
```

`cluster_real_yaw.py` imports from `cluster_pipeline.py` as a sibling — keep
them in the same directory.

---

## Known limitations

- Assumes a **flat ground plane** at target elevation; terrain relief introduces
  proportional position error.
- Assumes a **pinhole camera**; lens distortion degrades accuracy for detections
  near frame edges.
- Assumes the camera is **gimbal-stabilized to nadir with level roll**, and
  models only yaw as tracking the drone. A gimbal that doesn't fully null out
  the drone's pitch/roll (or a fixed, non-nadir camera mount) will bias every
  detection's ground offset — the pipeline has no way to detect this from the
  data alone.
- The verification classifier is **scale-sensitive**. It rejects valid targets
  much smaller than its training crops — a problem at high altitude or low
  resolution, not at A8 ≤150 ft. Disable it rather than fight it if target pixel
  size is low; geometric false-positive rejection works without it.
- **No optical zoom.** A two-stage search-then-verify concept using an optical
  zoom payload was evaluated but is not available; the pipeline runs
  single-stage, so clustering carries the full false-positive burden. This
  raises the value of survey overlap (more observations → larger ranking margin)
  and of the classifier gate at A8 pixel scales.
