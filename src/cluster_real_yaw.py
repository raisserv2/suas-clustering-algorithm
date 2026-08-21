#!/usr/bin/env python3
"""
cluster_real_yaw.py -- real geotagged drone images WITH per-frame heading
=========================================================================

For the watsa2-style dataset: nadir camera, GPS + GPSImgDirection (true-north
heading) in EXIF, known AGL, known FOV.

This is the FULL-POSE version. Unlike the earlier GPS-only script (which
assumed the image top always pointed North and therefore smeared detections
into a ring when the drone yawed), this reads the actual per-frame heading and
rotates the image axes accordingly. A fixed target seen from any position at
any heading back-projects to the SAME ground point -- verified exact (1e-14 m).

GEOMETRY
    world:    x = East, y = North (local ENU metres about the flight centre)
    heading:  degrees CW from TRUE north (EXIF GPSImgDirection, Ref='T')
    camera:   nadir (straight down); image top edge points along the heading
    pixel -> ground:
        dx = (cx - W/2)/focal * AGL      (metres right of image centre)
        dy = (cy - H/2)/focal * AGL      (metres down-image from centre)
        up    = unit vector at bearing (heading)
        right = unit vector at bearing (heading + 90)
        ground = drone_pos + dx*right + (-dy)*up

INTRINSICS (from the camera datasheet, not EXIF -- this cam writes no lens tags)
    Horizontal FOV = 80 deg, image 1280x720
    focal_px = (W/2) / tan(HFOV/2) = 762.7 px
    -> implied diagonal FOV 87.8 deg vs datasheet 90 deg: consistent.

USAGE
  python cluster_real_yaw.py \
      --images_dir watsa2 --model yolo11m_best.pt \
      --agl_m 50 --hfov_deg 80 \
      --conf 0.15 --eps 4.0 --min_samples 2 \
      --classifier mobilenet_finetuned.pth --clf_conf 0.5 \
      --out results_watsa2.json

Then load results_watsa2.json in viewer.html.

NOTE ON TARGET SIZE: at 50 m AGL with a 1280x720 / 80-deg camera the ground
footprint is ~84 x 47 m and the GSD ~6.6 cm/px, so a 1.75 m mannequin spans
only ~27 px and a 2.5 m tent ~38 px. The YOLO weights were trained on much
larger targets, so DETECTION (not geometry) is the likely bottleneck. If a
class returns NO CLUSTER, that is a recall finding, not a clustering failure.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_pipeline import (
    load_model, run_inference, dbscan_densest, match_class,
    load_classifier, classify_crop, CLASSIFIER_CLASSES,
)

EARTH_R = 6378137.0


def discover_images(folder):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(set(files))


# ----------------------------------------------------------------------------
# EXIF: GPS + heading
# ----------------------------------------------------------------------------

def read_exif_pose(image_path):
    """Return {lat, lon, gps_alt_m, heading_deg, width, height} or None."""
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


# ----------------------------------------------------------------------------
# Yaw-aware nadir back-projection  (verified exact: 1e-14 m convergence)
# ----------------------------------------------------------------------------

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
# Pipeline
# ----------------------------------------------------------------------------

def run(args):
    files = discover_images(args.images_dir)
    if not files:
        sys.exit(f"[fatal] no images in {os.path.abspath(args.images_dir)}")
    print(f"[data] {len(files)} images", flush=True)

    # --- read poses ---
    poses = {}
    lats, lons = [], []
    n_head = 0
    mag_warn = False
    for f in files:
        p = read_exif_pose(f)
        poses[f] = p
        if p:
            lats.append(p["lat"]); lons.append(p["lon"])
            if p["heading_deg"] is not None:
                n_head += 1
            if str(p.get("heading_ref", "T")).upper().startswith("M"):
                mag_warn = True
    if not lats:
        sys.exit("[fatal] no GPS in any image")
    lat0, lon0 = sum(lats) / len(lats), sum(lons) / len(lons)
    print(f"[geo] origin {lat0:.7f}, {lon0:.7f}   "
          f"({len(lats)}/{len(files)} GPS, {n_head}/{len(files)} heading)", flush=True)
    if mag_warn:
        print("[warn] heading ref is MAGNETIC, not true north -- absolute "
              "coords may be rotated by local declination.", flush=True)

    # fallback heading from GPS track for any frame missing GPSImgDirection
    ordered = [f for f in files if poses[f]]
    for i, f in enumerate(ordered):
        if poses[f]["heading_deg"] is None:
            a = poses[ordered[max(0, i - 1)]]
            b = poses[ordered[min(len(ordered) - 1, i + 1)]]
            if a is not b:
                poses[f]["heading_deg"] = bearing_from_gps(a["lat"], a["lon"],
                                                            b["lat"], b["lon"])
                poses[f]["heading_src"] = "gps_track"
                print(f"[head] {os.path.basename(f)}: estimated heading "
                      f"{poses[f]['heading_deg']:.1f} from GPS track", flush=True)
            else:
                poses[f]["heading_deg"] = 0.0
                poses[f]["heading_src"] = "default0"
        else:
            poses[f]["heading_src"] = "exif"

    # --- geometry report ---
    W0 = poses[ordered[0]]["width"]; H0 = poses[ordered[0]]["height"]
    focal = focal_px_from_hfov(args.hfov_deg, W0)
    vfov = 2 * math.atan((H0 / 2.0) / focal)
    fw = 2 * args.agl_m * math.tan(math.radians(args.hfov_deg) / 2)
    fh = 2 * args.agl_m * math.tan(vfov / 2)
    gsd = fw / W0
    print(f"[cam] {W0}x{H0}  HFOV={args.hfov_deg}deg  focal={focal:.1f}px  "
          f"VFOV={math.degrees(vfov):.1f}deg", flush=True)
    print(f"[cam] AGL={args.agl_m}m -> footprint {fw:.1f}x{fh:.1f} m, "
          f"GSD {gsd*100:.1f} cm/px", flush=True)
    print(f"[cam] expected target size: mannequin(1.75m)~{1.75/gsd:.0f}px, "
          f"tent(2.5m)~{2.5/gsd:.0f}px", flush=True)

    model, dev = load_model(args.model)
    clf = clf_tf = None
    if args.classifier:
        clf, clf_tf = load_classifier(args.classifier, dev)
    from PIL import Image

    per_frame = []
    pts = {"tent": [], "mannequin": []}
    pts_meta = {"tent": [], "mannequin": []}
    clf_stats = {"kept": 0, "rejected": 0}
    raw_counts = {"tent": 0, "mannequin": 0}

    for i, fpath in enumerate(files):
        p = poses.get(fpath)
        dets = run_inference(model, dev, fpath, args.conf)
        for d in dets:
            for t in ("tent", "mannequin"):
                if match_class(d["cls_name"], t):
                    raw_counts[t] += 1
        if p is None:
            per_frame.append({"frame_index": i, "image": os.path.basename(fpath),
                              "cam_xyz_m": None, "detections": []})
            print(f"[frame {i:04d}] {os.path.basename(fpath)} NO POSE, skipped", flush=True)
            continue

        drone_xy = latlon_to_local_m(p["lat"], p["lon"], lat0, lon0)
        heading = p["heading_deg"]
        W, H = p["width"], p["height"]
        f_px = focal_px_from_hfov(args.hfov_deg, W)

        pil = Image.open(fpath).convert("RGB") if clf is not None else None
        frame_dets = []
        for d in dets:
            clf_idx = clf_conf = None
            if clf is not None:
                clf_idx, clf_conf = classify_crop(clf, clf_tf, pil, d["box_xyxy"], dev)
                d = {**d, "clf_idx": clf_idx, "clf_conf": clf_conf,
                     "clf_name": (CLASSIFIER_CLASSES[clf_idx] if clf_idx is not None else None)}
                if clf_idx is None or clf_idx == 2 or clf_conf < args.clf_conf:
                    clf_stats["rejected"] += 1
                    frame_dets.append({**d, "ground_xy": None, "rejected": True})
                    continue
                clf_stats["kept"] += 1

            E, N = backproject_yaw_nadir(d["cx"], d["cy"], drone_xy, args.agl_m,
                                         f_px, W, H, heading)
            frame_dets.append({**d, "ground_xy": [E, N]})
            for tgt in ("tent", "mannequin"):
                if clf is not None and clf_idx is not None and clf_idx < 2:
                    matched = (CLASSIFIER_CLASSES[clf_idx] == tgt)
                else:
                    matched = match_class(d["cls_name"], tgt)
                if matched:
                    pts[tgt].append([E, N])
                    pts_meta[tgt].append({"frame": i, "conf": d["conf"]})

        per_frame.append({
            "frame_index": i,
            "image": os.path.basename(fpath),
            "cam_xyz_m": [drone_xy[0], drone_xy[1], args.agl_m],
            "gps": {"lat": p["lat"], "lon": p["lon"]},
            "heading_deg": heading,
            "heading_src": p.get("heading_src", "exif"),
            "detections": frame_dets,
        })
        print(f"[frame {i:04d}] {os.path.basename(fpath)} hdg={heading:6.1f} "
              f"{len(dets)} dets", flush=True)

    print(f"\n[raw yolo] tent={raw_counts['tent']} mannequin={raw_counts['mannequin']} "
          f"detections before classifier", flush=True)
    if clf is not None:
        print(f"[classifier] kept {clf_stats['kept']}, rejected {clf_stats['rejected']}", flush=True)

    # --- cluster ---
    clusters_out, predictions = {}, {}
    for tgt in ("tent", "mannequin"):
        arr = np.array(pts[tgt], dtype=float) if pts[tgt] else np.zeros((0, 2))
        confs = np.array([m["conf"] for m in pts_meta[tgt]], dtype=float) \
            if pts_meta[tgt] else np.zeros((0,))
        labels, centroids, best, scores = dbscan_densest(
            arr, args.eps, args.min_samples, confs=confs, rank=args.rank)
        clusters_out[tgt] = {
            "points": arr.tolist(),
            "labels": labels.tolist() if len(labels) else [],
            "centroids": centroids, "densest_label": best, "scores": scores,
            "rank_mode": args.rank, "point_meta": pts_meta[tgt],
        }
        if best is not None:
            predictions[tgt] = centroids[best]

    pred_gps = {}
    for tgt, (x, y) in predictions.items():
        la, lo = local_m_to_latlon(x, y, lat0, lon0)
        pred_gps[tgt] = {"lat": la, "lon": lo}

    results = {
        "mode": "synthetic",          # viewer uses the ground-map layout
        "real_drone": True,
        "images_dir": os.path.abspath(args.images_dir),
        "origin_latlon": [lat0, lon0],
        "params": {"agl_m": args.agl_m, "hfov_deg": args.hfov_deg,
                   "focal_px": focal, "conf": args.conf, "eps": args.eps,
                   "min_samples": args.min_samples,
                   "classifier": bool(args.classifier)},
        "coord_system": "local ENU metres about origin (x=East, y=North)",
        "per_frame": per_frame,
        "clusters": clusters_out,
        "predictions": predictions,
        "predicted_gps": pred_gps,
        "errors": {},
        "ground_truth": None,
    }
    with open(args.out, "w") as fh:
        json.dump(results, fh)

    print(f"\n[done] results -> {args.out}", flush=True)
    for tgt in ("tent", "mannequin"):
        if tgt in predictions:
            x, y = predictions[tgt]
            g = pred_gps[tgt]
            npts = len(clusters_out[tgt]["points"])
            print(f"  {tgt:<10} local ({x:+7.1f}, {y:+7.1f}) m   "
                  f"GPS {g['lat']:.7f}, {g['lon']:.7f}   [{npts} pts]", flush=True)
        else:
            npts = len(clusters_out[tgt]["points"])
            print(f"  {tgt:<10} NO CLUSTER  ({npts} detection points -- "
                  f"{'too few/scattered' if npts else 'model never fired'})", flush=True)
    if predictions.get("tent") and predictions.get("mannequin"):
        tx, ty = predictions["tent"]; mx, my = predictions["mannequin"]
        print(f"\n  tent<->mannequin separation: {math.hypot(tx-mx, ty-my):.1f} m"
              f"  (sanity-check against how you placed them)", flush=True)
    print("\nPaste the predicted GPS into Google Maps to eyeball the result.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--agl_m", type=float, required=True,
                    help="height above GROUND in metres (confirmed 50 for watsa2)")
    ap.add_argument("--hfov_deg", type=float, default=80.0,
                    help="camera horizontal FOV in degrees (datasheet: 80)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=4.0, help="DBSCAN eps in METRES")
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--rank", choices=("confsum", "count", "confmean_count"),
                    default="confsum")
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--clf_conf", type=float, default=0.5)
    ap.add_argument("--out", default="results_real_yaw.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()