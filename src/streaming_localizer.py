#!/usr/bin/env python3
"""
streaming_localizer.py -- online (frame-at-a-time) target localization
=====================================================================

Same pipeline as cluster_real_yaw.py, restructured for a live flight:

    DETECTION and GROUND PROJECTION run per frame, as each image arrives.
    CLUSTERING + RANKING run only at the end (or on demand for a live fix).

    drone sends image i  ->  loc.add_frame(image_i, pose_i)   # detect + project
    ...                                                        # repeat, in a loop
    landing / on demand  ->  loc.estimate()  /  loc.write_results()  # DBSCAN + rank

The final answer is identical to the batch script: DBSCAN over a set of
points does not depend on the order the points arrived in.

Two ways to drive it
--------------------
1. In-process, from your comms code (real flight) -- pose from autopilot
   telemetry, image as bytes / ndarray / PIL / path:

       loc = StreamingLocalizer("models/yolo11m_best.pt",
                                hfov_deg=80.2, agl_m=45.7,
                                classifier="models/mobilenet_finetuned.pth")
       for jpeg_bytes, tlm in link:                # your receive loop
           loc.add_frame(jpeg_bytes, {
               "lat": tlm.lat, "lon": tlm.lon,
               "agl_m": tlm.agl_m, "heading_deg": tlm.yaw_deg_true,
           })
           if loc._frame_counter % 20 == 0:
               print(loc.estimate())               # converging live fix
       loc.write_results("results.json")

2. Directory watcher (replay / testing) -- pose per image from an
   OpenDroneMap geo.txt sidecar if present (auto-detected in --images_dir or
   its parent, or pass --geo), otherwise from EXIF:

       python streaming_localizer.py \
           --images_dir incoming/ --model models/yolo11m_best.pt \
           --agl_m 45.7 --hfov_deg 80.2 \
           --classifier models/mobilenet_finetuned.pth \
           --out results.json --estimate_every 20

Differences vs cluster_real_yaw.py, by design
---------------------------------------------
* The local-frame origin is the FIRST usable frame's GPS, not the mean of
  every frame (the mean needs the whole flight up front). This only shifts
  the intermediate metre coordinates; the emitted GPS is unchanged to sub-mm.
* Heading is taken per frame from the pose you pass in (autopilot telemetry,
  geo.txt yaw_deg, or EXIF GPSImgDirection). A frame with no heading is
  recorded and skipped for geolocation, never guessed -- a wrong heading
  silently rotates that frame's detections onto the wrong ground point. The
  batch script's prev->next GPS-track estimate is not reproduced here.

cluster_real_yaw.py drives this same class in batch mode and pins the
origin to the mean, so its numeric output is unchanged.
"""

import argparse
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_pipeline import (          # noqa: E402
    load_model, run_inference, dbscan_densest, match_class,
    load_classifier, classify_crop, CLASSIFIER_CLASSES,
)

EARTH_R = 6378137.0

__all__ = [
    "StreamingLocalizer", "watch_directory", "print_final_report",
    "discover_images", "read_exif_pose", "read_geo_txt", "find_geo_txt",
    "pose_from_geo_record", "resolve_pose", "latlon_to_local_m",
    "local_m_to_latlon", "bearing_from_gps", "focal_px_from_hfov",
    "backproject_yaw_nadir", "EARTH_R",
]


# ----------------------------------------------------------------------------
# Image discovery + EXIF pose  (moved verbatim from cluster_real_yaw.py so the
# dependency runs core <- CLI, not the other way round)
# ----------------------------------------------------------------------------

def discover_images(folder):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(set(files))


def read_exif_pose(image_path):
    """Return {lat, lon, gps_alt_m, heading_deg, heading_ref, width, height}
    or None."""
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    try:
        img = Image.open(image_path)
        W, H = img.size
        exif = img._getexif() or {}
        tags = {TAGS.get(k, k): v for k, v in exif.items()}
        gps = tags.get("GPSInfo")
        if not gps:
            return None
        g = {GPSTAGS.get(k, k): v for k, v in gps.items()}

        def dms(t, ref):
            d, m, s = [float(x) for x in t]
            v = d + m / 60.0 + s / 3600.0
            return -v if ref in ("S", "W") else v

        if "GPSLatitude" not in g or "GPSLongitude" not in g:
            return None
        lat = dms(g["GPSLatitude"], g.get("GPSLatitudeRef", "N"))
        lon = dms(g["GPSLongitude"], g.get("GPSLongitudeRef", "E"))
        alt = float(g["GPSAltitude"]) if "GPSAltitude" in g else None

        heading = None
        if "GPSImgDirection" in g:
            heading = float(g["GPSImgDirection"])
            ref = str(g.get("GPSImgDirectionRef", "T"))
            # 'T' = true north (what we want). 'M' = magnetic -> would need
            # declination correction; we flag it rather than silently guess.
            if ref.upper().startswith("M"):
                heading = None if False else heading  # keep, but warn upstream
        return {"lat": lat, "lon": lon, "gps_alt_m": alt,
                "heading_deg": heading,
                "heading_ref": str(g.get("GPSImgDirectionRef", "T")),
                "width": W, "height": H}
    except Exception as e:
        print(f"[exif] {os.path.basename(image_path)}: {e}", flush=True)
        return None


# ----------------------------------------------------------------------------
# geo.txt sidecar (OpenDroneMap format) -- pose lives in a text file next to
# the images, not in EXIF:
#
#   EPSG:4326
#   # image_name longitude latitude altitude_amsl_m yaw_deg pitch_deg roll_deg [h_acc v_acc]
#   # yaw 0 = image top points north, pitch 0 = nadir  (ODM convention)
#   SUAS-..._020.jpg 80.23650729 12.99246149 62.97 14.29 2.49 0.38 0.71 0.66
#
# yaw_deg is exactly our heading (deg CW from true north, image top along the
# heading). pitch/roll are parsed but NOT used -- the pipeline assumes the
# gimbal holds nadir; the small residuals here (~2.5 deg pitch) are within that
# assumption. altitude is AMSL, so --agl_m is still supplied separately.
# ----------------------------------------------------------------------------

def read_geo_txt(path):
    """Parse an OpenDroneMap geo.txt. Returns
    {image_basename: {lat, lon, alt_amsl_m, heading_deg, pitch_deg, roll_deg}}.
    Raises ValueError if a projection header other than EPSG:4326 / WGS84
    lon-lat is declared (we can't reproject here)."""
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        return {}

    start = 0
    head = lines[0].upper().replace(" ", "")
    if head.startswith("EPSG:"):
        if head.split(":", 1)[1] not in ("4326",):
            raise ValueError(f"{path}: projection {lines[0]!r} unsupported "
                             f"(need EPSG:4326 lon/lat)")
        start = 1
    elif head.startswith("WGS84") and "UTM" not in head:
        start = 1
    elif "UTM" in head or head.startswith("PROJ") or head.startswith("+PROJ"):
        raise ValueError(f"{path}: projection {lines[0]!r} unsupported "
                         f"(need EPSG:4326 lon/lat)")

    poses = {}
    for ln in lines[start:]:
        if ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) < 4:
            continue
        try:
            lon, lat, alt = float(p[1]), float(p[2]), float(p[3])
        except ValueError:
            continue

        def _f(i):
            try:
                return float(p[i])
            except (IndexError, ValueError):
                return None

        poses[os.path.basename(p[0])] = {
            "lat": lat, "lon": lon, "alt_amsl_m": alt,
            "heading_deg": _f(4), "pitch_deg": _f(5), "roll_deg": _f(6),
        }
    return poses


def find_geo_txt(images_dir, explicit=None):
    """Locate a geo.txt: the explicit path if given, else geo.txt in the image
    folder or its parent. Returns a path or None."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    base = os.path.abspath(images_dir.rstrip("/") or ".")
    for c in (os.path.join(base, "geo.txt"),
              os.path.join(os.path.dirname(base), "geo.txt")):
        if os.path.isfile(c):
            return c
    return None


def _image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size          # (W, H) -- from the header, no decode


def pose_from_geo_record(g, image_path):
    """A geo.txt record -> the pose dict shape read_exif_pose returns."""
    W, H = _image_size(image_path)
    return {"lat": g["lat"], "lon": g["lon"], "gps_alt_m": g.get("alt_amsl_m"),
            "heading_deg": g.get("heading_deg"), "heading_ref": "T",
            "heading_src": "geo.txt", "width": W, "height": H}


def resolve_pose(image_path, geo_poses=None):
    """Pose for one image: geo.txt if it has an entry for this file, else EXIF,
    else None."""
    if geo_poses:
        g = geo_poses.get(os.path.basename(image_path))
        if g is not None:
            return pose_from_geo_record(g, image_path)
    p = read_exif_pose(image_path)
    if p is not None and p.get("heading_deg") is not None:
        p.setdefault("heading_src", "exif")
    return p


def latlon_to_local_m(lat, lon, lat0, lon0):
    """Equirectangular -> local ENU metres. x=East, y=North."""
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


def local_m_to_latlon(x, y, lat0, lon0):
    dlat = math.degrees(y / EARTH_R)
    dlon = math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    return lat0 + dlat, lon0 + dlon


def bearing_from_gps(lat1, lon1, lat2, lon2):
    """True bearing (deg CW from N) from point1 -> point2. Used only as a
    fallback if a frame is missing GPSImgDirection."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def focal_px_from_hfov(hfov_deg, image_width_px):
    return (image_width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def backproject_yaw_nadir(cx, cy, drone_xy, agl_m, focal_px, W, H, heading_deg):
    """Pixel -> ground (East, North) for a NADIR camera yawed to `heading_deg`.

    The image's top edge points along the heading; +x (right) points at
    heading+90. Ground offsets are the pixel offsets scaled by AGL/focal, then
    rotated into world axes by the heading.
    """
    dx = (cx - W / 2.0) / focal_px * agl_m     # metres right of centre
    dy = (cy - H / 2.0) / focal_px * agl_m     # metres DOWN-image from centre
    h = math.radians(heading_deg)
    up_E, up_N = math.sin(h), math.cos(h)                     # bearing h
    rt_E, rt_N = math.sin(h + math.pi / 2), math.cos(h + math.pi / 2)
    E = drone_xy[0] + dx * rt_E + (-dy) * up_E
    N = drone_xy[1] + dx * rt_N + (-dy) * up_N
    return E, N


# ----------------------------------------------------------------------------
# The streaming core
# ----------------------------------------------------------------------------

class StreamingLocalizer:
    """Frame-at-a-time detector + geolocator with deferred clustering.

    add_frame()      -- call once per incoming image (detect + project). Cheap
                        relative to the clustering, and independent of every
                        other frame, so it runs happily in a receive loop.
    estimate()       -- cluster + rank the points accumulated SO FAR. Call it
                        on a timer for a live fix that converges as frames come
                        in, and once after the last frame for the drop.
    build_results()  -- full results.json dict (same schema as the batch script).
    write_results()  -- atomic dump of build_results(); use it as a checkpoint.

    Thread model: call add_frame() from a single thread. estimate() /
    build_results() / write_results() may be called from another thread at any
    time -- they take a consistent snapshot under a lock.
    """

    def __init__(self, model, hfov_deg, agl_m, *,
                 classifier=None, clf_conf=0.5, conf=0.15,
                 eps=4.0, min_samples=2, rank="confsum",
                 origin_latlon=None, images_dir=None):
        self.model, self.dev = load_model(model)
        self.clf = self.clf_tf = None
        if classifier:
            self.clf, self.clf_tf = load_classifier(classifier, self.dev)

        self.hfov_deg = float(hfov_deg)
        self.agl_m = float(agl_m)
        self.conf = float(conf)
        self.clf_conf = float(clf_conf)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.rank = rank
        self.origin = tuple(origin_latlon) if origin_latlon else None
        self.images_dir = images_dir

        self.focal_px = None
        self.pts = {"tent": [], "mannequin": []}
        self.meta = {"tent": [], "mannequin": []}
        self.per_frame = []
        self.raw_counts = {"tent": 0, "mannequin": 0}
        self.clf_stats = {"kept": 0, "rejected": 0}

        self._frame_counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    @staticmethod
    def _as_pil(image):
        """image -> PIL RGB. Accepts a path, PIL image, raw bytes, or an
        HxWx3 RGB ndarray."""
        from PIL import Image
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return Image.open(image).convert("RGB")

    # ------------------------------------------------------------------

    def add_frame(self, image, pose, frame_index=None, name=None):
        """Detect + project one frame. Returns the per-frame record (which is
        also appended to self.per_frame).

        image : path str | PIL.Image | bytes | HxWx3 RGB ndarray
        pose  : dict with
                  lat, lon                 (required)
                  heading_deg              (deg CW from true north; required --
                                            a frame without it is recorded and
                                            skipped for geolocation, never
                                            guessed)
                  agl_m                    (optional; falls back to ctor agl_m)
                  width, height            (optional; else read from the image)
                  heading_src              (optional label kept in the output)
                or None if telemetry for this frame is missing (the frame is
                still run through the detector for the raw-count tally, then
                recorded as pose-less and skipped for geolocation).
        """
        from PIL import Image

        with self._lock:
            fidx = self._frame_counter if frame_index is None else frame_index
            self._frame_counter = fidx + 1

        is_path = isinstance(image, str)
        fname = name or (os.path.basename(image) if is_path
                         else f"frame_{fidx:04d}")

        pil = None
        if self.clf is not None or not is_path:
            pil = self._as_pil(image)
        infer_src = image if is_path else pil

        dets = run_inference(self.model, self.dev, infer_src, self.conf)
        for d in dets:
            for t in ("tent", "mannequin"):
                if match_class(d["cls_name"], t):
                    with self._lock:
                        self.raw_counts[t] += 1

        if pose is None:
            rec = {"frame_index": fidx, "image": fname,
                   "cam_xyz_m": None, "detections": []}
            with self._lock:
                self.per_frame.append(rec)
            return rec

        lat, lon = float(pose["lat"]), float(pose["lon"])

        # Heading is required per frame and is never guessed: a wrong heading
        # rotates this frame's detections onto the wrong ground point and
        # quietly poisons the cluster. No heading -> record it and skip.
        heading = pose.get("heading_deg")
        heading_src = pose.get("heading_src") or "exif"
        if heading is None:
            print(f"[warn] frame {fidx} ({fname}): GPS but no heading "
                  f"(GPSImgDirection) -- not geolocated", flush=True)
            rec = {"frame_index": fidx, "image": fname, "cam_xyz_m": None,
                   "detections": [], "no_heading": True}
            with self._lock:
                self.per_frame.append(rec)
            return rec

        with self._lock:
            if self.origin is None:
                self.origin = (lat, lon)
            origin = self.origin

        agl = float(pose.get("agl_m") or self.agl_m)

        if pose.get("width") and pose.get("height"):
            W, H = int(pose["width"]), int(pose["height"])
        elif pil is not None:
            W, H = pil.size
        else:
            with Image.open(image) as im:
                W, H = im.size

        if self.focal_px is None:
            self.focal_px = focal_px_from_hfov(self.hfov_deg, W)
        f_px = focal_px_from_hfov(self.hfov_deg, W)
        drone_xy = latlon_to_local_m(lat, lon, origin[0], origin[1])

        if self.clf is not None and pil is None:
            pil = self._as_pil(image)

        frame_dets = []
        new_pts = {"tent": [], "mannequin": []}
        for d in dets:
            clf_idx = clf_conf = None
            if self.clf is not None:
                clf_idx, clf_conf = classify_crop(self.clf, self.clf_tf, pil,
                                                  d["box_xyxy"], self.dev)
                d = {**d, "clf_idx": clf_idx, "clf_conf": clf_conf,
                     "clf_name": (CLASSIFIER_CLASSES[clf_idx]
                                  if clf_idx is not None else None)}
                if clf_idx is None or clf_idx == 2 or clf_conf < self.clf_conf:
                    with self._lock:
                        self.clf_stats["rejected"] += 1
                    frame_dets.append({**d, "ground_xy": None, "rejected": True})
                    continue
                with self._lock:
                    self.clf_stats["kept"] += 1

            E, N = backproject_yaw_nadir(d["cx"], d["cy"], drone_xy, agl,
                                         f_px, W, H, heading)
            frame_dets.append({**d, "ground_xy": [E, N]})
            for tgt in ("tent", "mannequin"):
                if self.clf is not None and clf_idx is not None and clf_idx < 2:
                    matched = (CLASSIFIER_CLASSES[clf_idx] == tgt)
                else:
                    matched = match_class(d["cls_name"], tgt)
                if matched:
                    new_pts[tgt].append((E, N, d["conf"]))

        rec = {
            "frame_index": fidx,
            "image": fname,
            "cam_xyz_m": [drone_xy[0], drone_xy[1], agl],
            "gps": {"lat": lat, "lon": lon},
            "heading_deg": heading,
            "heading_src": heading_src,
            "detections": frame_dets,
        }
        with self._lock:
            for tgt in ("tent", "mannequin"):
                for (E, N, c) in new_pts[tgt]:
                    self.pts[tgt].append([E, N])
                    self.meta[tgt].append({"frame": fidx, "conf": c})
            self.per_frame.append(rec)
        return rec

    # ------------------------------------------------------------------

    def _cluster_now(self):
        """DBSCAN + rank the points accumulated so far. Pure function of the
        current point set -- safe to call as often as you like."""
        with self._lock:
            snap_pts = {t: list(v) for t, v in self.pts.items()}
            snap_meta = {t: list(v) for t, v in self.meta.items()}
            origin = self.origin

        clusters_out, predictions, pred_gps = {}, {}, {}
        for tgt in ("tent", "mannequin"):
            arr = (np.array(snap_pts[tgt], dtype=float)
                   if snap_pts[tgt] else np.zeros((0, 2)))
            confs = (np.array([m["conf"] for m in snap_meta[tgt]], dtype=float)
                     if snap_meta[tgt] else np.zeros((0,)))
            labels, centroids, best, scores = dbscan_densest(
                arr, self.eps, self.min_samples, confs=confs, rank=self.rank)
            clusters_out[tgt] = {
                "points": arr.tolist(),
                "labels": labels.tolist() if len(labels) else [],
                "centroids": centroids, "densest_label": best, "scores": scores,
                "rank_mode": self.rank, "point_meta": snap_meta[tgt],
            }
            if best is not None:
                predictions[tgt] = centroids[best]
                if origin is not None:
                    la, lo = local_m_to_latlon(centroids[best][0],
                                               centroids[best][1],
                                               origin[0], origin[1])
                    pred_gps[tgt] = {"lat": la, "lon": lo}
        return clusters_out, predictions, pred_gps

    def estimate(self):
        """Current best GPS per class, plus the numbers you'd watch to decide
        whether the fix is trustworthy yet."""
        clusters_out, _, pred_gps = self._cluster_now()
        out = {}
        for tgt in ("tent", "mannequin"):
            cl = clusters_out[tgt]
            best = cl["densest_label"]
            if best is None:
                out[tgt] = {"located": False, "n_points": len(cl["points"])}
            else:
                out[tgt] = {
                    "located": True,
                    "local_xy": cl["centroids"][best],
                    "gps": pred_gps.get(tgt),
                    "n_points": len(cl["points"]),
                    "n_clusters": len(cl["scores"]),
                    "score": cl["scores"].get(best),
                }
        return out

    def build_results(self):
        """Full results.json dict -- same schema cluster_real_yaw.py emits, so
        it loads in tools/viewer.html unchanged."""
        clusters_out, predictions, pred_gps = self._cluster_now()
        with self._lock:
            per_frame = list(self.per_frame)
            origin = list(self.origin) if self.origin else None
        return {
            "mode": "synthetic",          # viewer uses the ground-map layout
            "real_drone": True,
            "images_dir": self.images_dir,
            "origin_latlon": origin,
            "params": {"agl_m": self.agl_m, "hfov_deg": self.hfov_deg,
                       "focal_px": self.focal_px, "conf": self.conf,
                       "eps": self.eps, "min_samples": self.min_samples,
                       "classifier": self.clf is not None},
            "coord_system": "local ENU metres about origin (x=East, y=North)",
            "per_frame": per_frame,
            "clusters": clusters_out,
            "predictions": predictions,
            "predicted_gps": pred_gps,
            "errors": {},
            "ground_truth": None,
        }

    def write_results(self, path):
        """Atomic dump of build_results(). Safe to call mid-flight as a
        checkpoint -- a crash leaves the last good file intact."""
        results = self.build_results()
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(results, fh)
        os.replace(tmp, path)
        return results


# ----------------------------------------------------------------------------
# Reporting helpers (shared with cluster_real_yaw.py so the console output of
# the two entry points can't drift apart)
# ----------------------------------------------------------------------------

def print_final_report(results, out_path):
    predictions = results.get("predictions", {})
    pred_gps = results.get("predicted_gps", {})
    clusters = results.get("clusters", {})
    print(f"\n[done] results -> {out_path}", flush=True)
    for tgt in ("tent", "mannequin"):
        npts = len(clusters.get(tgt, {}).get("points", []))
        if tgt in predictions:
            x, y = predictions[tgt]
            g = pred_gps.get(tgt, {"lat": float("nan"), "lon": float("nan")})
            print(f"  {tgt:<10} local ({x:+7.1f}, {y:+7.1f}) m   "
                  f"GPS {g['lat']:.7f}, {g['lon']:.7f}   [{npts} pts]", flush=True)
        else:
            print(f"  {tgt:<10} NO CLUSTER  ({npts} detection points -- "
                  f"{'too few/scattered' if npts else 'model never fired'})",
                  flush=True)
    if predictions.get("tent") and predictions.get("mannequin"):
        tx, ty = predictions["tent"]
        mx, my = predictions["mannequin"]
        print(f"\n  tent<->mannequin separation: {math.hypot(tx-mx, ty-my):.1f} m"
              f"  (sanity-check against how you placed them)", flush=True)
    print("\nPaste the predicted GPS into Google Maps to eyeball the result.",
          flush=True)


def _print_estimate(est):
    bits = []
    for tgt in ("tent", "mannequin"):
        e = est.get(tgt, {})
        if e.get("located"):
            g = e.get("gps") or {}
            bits.append(f"{tgt} {g.get('lat', float('nan')):.6f},"
                        f"{g.get('lon', float('nan')):.6f} "
                        f"[{e['n_points']}pt/{e['n_clusters']}cl "
                        f"score {e['score']:.2f}]")
        else:
            bits.append(f"{tgt} -- [{e.get('n_points', 0)}pt]")
    print("  [live] " + "   ".join(bits), flush=True)


# ----------------------------------------------------------------------------
# Directory-watch driver: process images as they land in a folder
# ----------------------------------------------------------------------------

def watch_directory(loc, folder, *, geo_poses=None, poll_sec=1.0, stable_sec=0.4,
                    idle_timeout=20.0, estimate_every=0, checkpoint_every=25,
                    out_path=None, _preseen=None):
    """Poll `folder`; for every new image whose size has stopped changing,
    resolve its pose (geo.txt entry if `geo_poses` has one, else EXIF) and hand
    it to loc.add_frame(). Returns the number of frames fed. Stops after
    `idle_timeout` s with no new image (None = run until KeyboardInterrupt)."""
    seen = set(_preseen or [])
    fidx = 0
    last_activity = time.time()
    print(f"[watch] {os.path.abspath(folder)}  (poll {poll_sec}s, "
          f"{'run until Ctrl-C' if idle_timeout is None else f'stop after {idle_timeout:.0f}s idle'})",
          flush=True)
    while True:
        candidates = [f for f in discover_images(folder) if f not in seen]
        ready = []
        for f in candidates:
            try:
                s1 = os.path.getsize(f)
            except OSError:
                continue
            time.sleep(stable_sec)
            try:
                s2 = os.path.getsize(f)
            except OSError:
                continue
            if s1 == s2 and s1 > 0:
                ready.append(f)

        for f in sorted(ready):
            seen.add(f)
            pose = resolve_pose(f, geo_poses)
            rec = loc.add_frame(f, pose, frame_index=fidx)
            fidx += 1
            last_activity = time.time()
            if rec.get("cam_xyz_m") is None:
                why = "NO HEADING" if rec.get("no_heading") else "NO POSE"
                print(f"[frame {rec['frame_index']:04d}] {os.path.basename(f)} "
                      f"{why}, skipped", flush=True)
            else:
                print(f"[frame {rec['frame_index']:04d}] {os.path.basename(f)} "
                      f"hdg={rec['heading_deg']:6.1f} ({rec['heading_src']}) "
                      f"{len(rec['detections'])} dets", flush=True)
            if estimate_every and fidx % estimate_every == 0:
                _print_estimate(loc.estimate())
            if out_path and checkpoint_every and fidx % checkpoint_every == 0:
                loc.write_results(out_path)
                print(f"[checkpoint] {fidx} frames -> {out_path}", flush=True)

        if idle_timeout is not None and (time.time() - last_activity) > idle_timeout:
            print(f"[watch] no new images for {idle_timeout:.0f}s -- finalizing",
                  flush=True)
            break
        time.sleep(poll_sec)
    return fidx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True,
                    help="folder the drone drops images into (watched live)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--agl_m", type=float, required=True,
                    help="height above GROUND in metres")
    ap.add_argument("--hfov_deg", type=float, default=80.2,
                    help="camera horizontal FOV in degrees (A8 mini: 80.2)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=4.0, help="DBSCAN eps in METRES")
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--rank", choices=("confsum", "count", "confmean_count"),
                    default="confsum")
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--clf_conf", type=float, default=0.5)
    ap.add_argument("--geo", default=None,
                    help="path to an OpenDroneMap geo.txt (pose per image). "
                         "Default: auto-detect geo.txt in --images_dir or its "
                         "parent. Falls back to EXIF for images not listed.")
    ap.add_argument("--out", default="results_stream.json")
    ap.add_argument("--poll_sec", type=float, default=1.0)
    ap.add_argument("--stable_sec", type=float, default=0.4,
                    help="require file size unchanged this long before reading "
                         "(guards against a half-written image)")
    ap.add_argument("--idle_timeout", type=float, default=20.0,
                    help="finalize after this many seconds with no new image; "
                         "<=0 runs until Ctrl-C")
    ap.add_argument("--estimate_every", type=int, default=0,
                    help="print a converging live target fix every N frames "
                         "(0 = off)")
    ap.add_argument("--checkpoint_every", type=int, default=25,
                    help="rewrite --out every N frames so a crash keeps a "
                         "partial result (0 = off)")
    ap.add_argument("--process_existing", action="store_true",
                    help="also process images already in the folder at startup")
    args = ap.parse_args()

    loc = StreamingLocalizer(
        args.model, hfov_deg=args.hfov_deg, agl_m=args.agl_m,
        classifier=args.classifier, clf_conf=args.clf_conf, conf=args.conf,
        eps=args.eps, min_samples=args.min_samples, rank=args.rank,
        images_dir=os.path.abspath(args.images_dir),
    )
    print(f"[cfg] HFOV={args.hfov_deg}deg  AGL={args.agl_m}m  conf={args.conf} "
          f"eps={args.eps} min_samples={args.min_samples} rank={args.rank}  "
          f"classifier={'on' if args.classifier else 'off'}", flush=True)

    geo_poses = None
    geo_path = find_geo_txt(args.images_dir, args.geo)
    if geo_path:
        geo_poses = read_geo_txt(geo_path)
        print(f"[geo] pose source: {geo_path}  ({len(geo_poses)} images; "
              f"yaw=heading, pitch/roll ignored -- gimbal-nadir assumed)",
              flush=True)
    elif args.geo:
        sys.exit(f"[fatal] --geo {args.geo} not found")
    else:
        print("[geo] no geo.txt found -- reading pose from image EXIF", flush=True)

    preseen = set()
    if not args.process_existing:
        preseen = set(discover_images(args.images_dir))
        if preseen:
            print(f"[watch] ignoring {len(preseen)} image(s) already present "
                  f"(--process_existing to include them)", flush=True)

    try:
        watch_directory(
            loc, args.images_dir, geo_poses=geo_poses,
            poll_sec=args.poll_sec, stable_sec=args.stable_sec,
            idle_timeout=(None if args.idle_timeout <= 0 else args.idle_timeout),
            estimate_every=args.estimate_every,
            checkpoint_every=args.checkpoint_every, out_path=args.out,
            _preseen=preseen,
        )
    except KeyboardInterrupt:
        print("\n[watch] interrupted -- finalizing with what we have", flush=True)

    print(f"\n[raw yolo] tent={loc.raw_counts['tent']} "
          f"mannequin={loc.raw_counts['mannequin']} detections before classifier",
          flush=True)
    if loc.clf is not None:
        print(f"[classifier] kept {loc.clf_stats['kept']}, "
              f"rejected {loc.clf_stats['rejected']}", flush=True)

    results = loc.write_results(args.out)
    print_final_report(results, args.out)


if __name__ == "__main__":
    main()
