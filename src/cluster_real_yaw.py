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

BATCH vs STREAMING
    This script is the BATCH entry point: it needs the whole flight folder up
    front (it averages every frame's GPS to pick the local-frame origin, and
    fills any missing heading from the bearing between the neighbouring frames).
    It runs the per-frame work through StreamingLocalizer and only defers the
    origin/heading passes that genuinely need all frames.

    For a LIVE flight -- process each image as it arrives, cluster only at the
    end -- use src/streaming_localizer.py, which drives the same
    StreamingLocalizer frame by frame.

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
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_localizer import (  # noqa: E402  (helpers re-exported for compat)
    StreamingLocalizer, print_final_report,
    discover_images, read_exif_pose, latlon_to_local_m, local_m_to_latlon,
    bearing_from_gps, focal_px_from_hfov, backproject_yaw_nadir, EARTH_R,
)


def run(args):
    files = discover_images(args.images_dir)
    if not files:
        sys.exit(f"[fatal] no images in {os.path.abspath(args.images_dir)}")
    print(f"[data] {len(files)} images", flush=True)

    # --- read poses (batch: from EXIF) ---
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

    # BATCH-ONLY pass 1: local-frame origin = mean of every frame's GPS.
    # (The streaming entry point uses the first frame instead.)
    lat0, lon0 = sum(lats) / len(lats), sum(lons) / len(lons)
    print(f"[geo] origin {lat0:.7f}, {lon0:.7f}   "
          f"({len(lats)}/{len(files)} GPS, {n_head}/{len(files)} heading)", flush=True)
    if mag_warn:
        print("[warn] heading ref is MAGNETIC, not true north -- absolute "
              "coords may be rotated by local declination.", flush=True)

    # BATCH-ONLY pass 2: fill any missing heading from the bearing between the
    # PREVIOUS and NEXT frame (needs both neighbours -> can't be done live).
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

    # --- per-frame work: detect + project, one frame at a time ---
    loc = StreamingLocalizer(
        args.model, hfov_deg=args.hfov_deg, agl_m=args.agl_m,
        classifier=args.classifier, clf_conf=args.clf_conf, conf=args.conf,
        eps=args.eps, min_samples=args.min_samples, rank=args.rank,
        origin_latlon=(lat0, lon0),      # pinned: keeps batch output unchanged
        images_dir=os.path.abspath(args.images_dir),
    )

    for i, fpath in enumerate(files):
        p = poses.get(fpath)
        pose = None if p is None else {
            "lat": p["lat"], "lon": p["lon"],
            "heading_deg": p["heading_deg"],
            "heading_src": p.get("heading_src", "exif"),
            "width": p["width"], "height": p["height"],
            "agl_m": args.agl_m,
        }
        rec = loc.add_frame(fpath, pose, frame_index=i)
        if pose is None:
            print(f"[frame {i:04d}] {os.path.basename(fpath)} NO POSE, skipped",
                  flush=True)
        else:
            print(f"[frame {i:04d}] {os.path.basename(fpath)} "
                  f"hdg={rec['heading_deg']:6.1f} {len(rec['detections'])} dets",
                  flush=True)

    print(f"\n[raw yolo] tent={loc.raw_counts['tent']} "
          f"mannequin={loc.raw_counts['mannequin']} detections before classifier",
          flush=True)
    if loc.clf is not None:
        print(f"[classifier] kept {loc.clf_stats['kept']}, "
              f"rejected {loc.clf_stats['rejected']}", flush=True)

    # --- deferred stage: cluster + rank everything accumulated ---
    results = loc.build_results()
    with open(args.out, "w") as fh:
        json.dump(results, fh)
    print_final_report(results, args.out)


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
