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

### Entry points

| Script | Use for | Pose source |
|---|---|---|
| `src/cluster_real_yaw.py` | **real flights, post-flight** (whole folder) | EXIF GPS + `GPSImgDirection` heading + `--agl_m` |
| `src/streaming_localizer.py` | **live flights** (frame-at-a-time) | per-frame pose from autopilot telemetry, or EXIF via the folder watcher |
| `src/cluster_pipeline.py synthetic` | pre-rendered synthetic flights | `flight_log.jsonl` camera poses; reports error against known ground truth |

All share the same stages 1–5; only the pose source and *when* clustering runs
differ. The synthetic mode is what produced the 3.80 / 3.94 m validation figures,
since it has ground truth to measure against.

### Streaming mode

`cluster_real_yaw.py` waits for the whole flight before it does anything. On the
aircraft the images arrive one at a time, and stages **1 (detect)** and
**3 (geolocate)** have no dependency on any other frame — only stage **4
(cluster)** does. `src/streaming_localizer.py` runs 1–3 per frame as images
land, accumulates the ground points, and runs 4–5 only on demand:

```
loc.add_frame(image, pose)   # detect + project — call in your receive loop
...                          # repeat per image
loc.estimate()               # cluster the points so far — a converging live fix
loc.write_results("out.json")# final answer + viewer file
```

DBSCAN over a point set does not depend on arrival order, so the streamed final
answer is identical to the batch script's. It differs only in two respects,
both harmless:

- **Local-frame origin** is the first usable frame's GPS, not the mean of all
  frames (the mean needs the whole flight up front). Shifts the intermediate
  metre coordinates only; emitted GPS is unchanged to sub-mm.
- **Heading is required per frame** — from `pose["heading_deg"]` (autopilot
  telemetry) or EXIF `GPSImgDirection` via the watcher. A frame without it is
  logged and skipped for geolocation, never guessed. The batch script's
  prev→next GPS-track estimate is not reproduced here.

Run it as a **folder watcher** (images written into a directory by the ground
station) — see *Quick start* — or drive it **frame by frame from your comms
loop** — see *Integration*.

### Dry-running it across two machines

`tools/feed_flight.py` replays a folder as a drone would: it sends images one
at a time, in filename order, with a fixed gap, to the incoming dir the watcher
reads. SRC and DST may each be local or `user@host:/path` (over ssh + rsync).

**Pose source for the test.** The watcher takes pose from an OpenDroneMap
`geo.txt` (columns `image_name longitude latitude altitude_amsl_m yaw_deg
pitch_deg roll_deg …`; `yaw_deg` is the heading, pitch/roll are ignored under
the gimbal-nadir assumption) when one is present in `--images_dir` or its
parent — otherwise it falls back to image EXIF. `feed_flight.py` ships a
`geo.txt` found beside SRC before the first image, so the watcher has pose from
frame 0. This is test-path only; the real-drone flow (`cluster_real_yaw.py`,
and `StreamingLocalizer.add_frame` with telemetry pose) is unchanged.

On the **pipeline machine** — start the watcher first, incoming dir empty:

```bash
mkdir -p ~/incoming && rm -f ~/incoming/*
python src/streaming_localizer.py \
    --images_dir ~/incoming \
    --model models/yolo11m_best.pt \
    --agl_m 45.7 --hfov_deg 80.2 \
    --conf 0.15 --eps 4.0 --min_samples 2 \
    --classifier models/mobilenet_finetuned.pth \
    --out results.json --estimate_every 5 --idle_timeout 30
```

On the **machine holding the images** (needs only `tools/feed_flight.py` + ssh
to the pipeline machine):

```bash
python tools/feed_flight.py \
    /path/to/flight_images  user@<pipeline-host>:~/incoming  -i 2
```

The watcher prints `[geo] pose source: …`, then `[frame …] N dets` per image, a
`[live]` fix every `--estimate_every`, a `[checkpoint]` every
`--checkpoint_every`, then the final block and `results.json` once no new image
has arrived for `--idle_timeout` s (keep that comfortably above the 2 s gap;
raise it if the pipeline machine is CPU-only and a backlog builds). A frame
with no `geo.txt` entry and no EXIF heading is logged and skipped. Same-machine
test: use two terminals and a local `./incoming` path on both sides — point the
feeder at the folder that has `geo.txt` in it.

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

**1. Check the flight metadata is usable** — before any new flight or camera:

```bash
python src/inspect_dataset.py /path/to/flight_images
```

Reports GPS, per-frame heading, altitude, intrinsics, and prints a verdict.
If it says heading is missing, **stop** — see *Data requirements*.

**2. Localize.** Same maths and same `results.json` either way; pick by *when*
you have the images. `--agl_m` is height above **ground** in metres
(150 ft = 45.7 m), not GPS sea-level altitude.

*Post-flight* — you have the whole folder (A8 mini at 150 ft):

```bash
python src/cluster_real_yaw.py \
    --images_dir /path/to/flight_images \
    --model models/yolo11m_best.pt \
    --agl_m 45.7 --hfov_deg 80.2 \
    --conf 0.15 --eps 4.0 --min_samples 2 \
    --classifier models/mobilenet_finetuned.pth \
    --out results.json
```

*In flight* — detect + geolocate each image as it lands, cluster at the end.
Point it at a directory the ground station drops images into:

```bash
python src/streaming_localizer.py \
    --images_dir /path/to/incoming \
    --model models/yolo11m_best.pt \
    --agl_m 45.7 --hfov_deg 80.2 \
    --conf 0.15 --eps 4.0 --min_samples 2 \
    --classifier models/mobilenet_finetuned.pth \
    --out results.json --estimate_every 20
```

It picks up each new image once its size settles, prints a converging fix every
`--estimate_every` frames, rewrites `--out` every `--checkpoint_every` frames
(default 25) so a crash keeps a partial, and finalizes after `--idle_timeout` s
with no new image (default 20; `<=0` runs until Ctrl-C). Heading is read from
each image's EXIF `GPSImgDirection`; a frame without it is skipped with a
warning. To feed frames straight from your comms loop instead of a folder, see
*Integration*.

Output (both):

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

`src/cluster_real_yaw.py` and `src/streaming_localizer.py` take all of the
above identically. The streaming script adds watcher controls:
`--estimate_every` (print a live fix every N frames), `--checkpoint_every`
(rewrite `--out` every N frames, default 25), `--idle_timeout` (finalize after
N idle seconds, default 20; `<=0` = until Ctrl-C), `--stable_sec` (wait for an
image's size to stop changing before reading it), `--process_existing` (also
process images already in the folder at startup).

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

### Post-flight, from Python

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

### In flight, frame by frame

Feed images straight from your receive loop — no watched folder. `add_frame`
does detection + projection (cheap, per frame, independent of other frames);
clustering + ranking are deferred to `estimate()` / `write_results()`.

```python
from src.streaming_localizer import StreamingLocalizer

loc = StreamingLocalizer(
    "models/yolo11m_best.pt", hfov_deg=80.2, agl_m=45.7,
    classifier="models/mobilenet_finetuned.pth",
    conf=0.15, eps=4.0, min_samples=2, rank="confsum",
)

for jpeg_bytes, tlm in link:                  # your comms loop
    loc.add_frame(jpeg_bytes, {
        "lat": tlm.lat, "lon": tlm.lon,
        "agl_m": tlm.agl_m,                    # optional; else the ctor value
        "heading_deg": tlm.yaw_deg_true,       # REQUIRED, deg CW from true north
    })
    if loc._frame_counter % 20 == 0:
        print(loc.estimate())                 # converging live fix (optional)

loc.write_results("results.json")             # final answer + viewer file
```

- `image` may be a path, `bytes`, a PIL image, or an `HxWx3` RGB ndarray.
- `pose` must carry `lat`, `lon`, and `heading_deg`. A frame with
  `heading_deg=None`, or `pose=None`, is logged and skipped for geolocation —
  never projected with a guessed heading.
- `estimate()` returns, per class, the current GPS plus `n_points` /
  `n_clusters` / `score` so you can gate the drop on a fix that has converged.
- Call `add_frame` from **one** thread; `estimate()` / `write_results()` may be
  called from another at any time (they snapshot under a lock).

### Reading results

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
                 cam_xyz_m = null -> frame skipped (no pose, or no heading)
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
src/streaming_localizer.py        per-frame detect+project core; live entry point
src/cluster_real_yaw.py           real-flight batch entry point (EXIF GPS + heading)  <- run this post-flight
src/inspect_dataset.py            metadata validator — run before any new flight
tools/viewer.html                 replay UI, loads results.json, no GPU needed
tools/feed_flight.py              replay an image folder as a 1-at-a-time drone stream (for testing streaming mode)
tools/benchmark_quantization.py   quantized-variant comparison
models/                           detector + verification classifier weights
```

`cluster_real_yaw.py` imports `streaming_localizer.py`, which imports
`cluster_pipeline.py` — all three are siblings, keep them in the same directory.

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
