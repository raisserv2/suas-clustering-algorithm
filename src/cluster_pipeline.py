#!/usr/bin/env python3
"""
cluster_pipeline.py  --  detection -> geolocation -> cluster -> rank
====================================================================

Tests the clustering hypothesis on a flight sequence:

    Feed numbered frames to YOLO in flight order. For each detection,
    back-project the box centre to a GROUND (X, Y) coordinate using the known
    camera pose. Accumulate all (X, Y) points per class across the flight,
    DBSCAN them, take the DENSEST cluster per class as the predicted target
    location, and compare to ground truth.

Works on TWO data sources:

  (A) SYNTHETIC flight from generate_flight.py
      - exact nadir back-projection from cam pose + altitude + intrinsics
      - has ground_truth.json -> reports centroid error in metres

  (B) REAL geotagged GoPro images
      - Tier-1 geolocation: each detection inherits the image's GPS coord
        (lat/lon from EXIF). Crude but enough for a first clustering test.
      - no ground truth -> just reports the predicted clusters

OUTPUT
    results.json   self-contained replay file consumed by the viewer:
                   per-frame detections (pixel boxes), geolocated points,
                   running + final clusters, centroids, error metrics.

USAGE
  Synthetic:
    python cluster_pipeline.py synthetic \
        --flight_dir flight_out --model best.pt \
        --conf 0.15 --eps 2.0 --min_samples 3 --out results.json

  Real GoPro (Tier-1, GPS-per-image):
    python cluster_pipeline.py real \
        --images_dir gopro_imgs --model best.pt \
        --conf 0.15 --eps_m 5.0 --min_samples 2 --out results_real.json

NOTE on conf: we deliberately run a LOW confidence threshold. Aggregation /
clustering is what rejects false positives (scattered points don't form dense
clusters), so we WANT many true-positive points feeding the cluster. Don't
raise conf to suppress FPs here -- let DBSCAN do it.
"""

import argparse
import json
import math
import os
import sys
import glob

import numpy as np


# ----------------------------------------------------------------------------
# Optional MobileNetV3 classifier wrapper (matches the training notebook)
#   - MobileNetV3-Large, 3 classes: ['tent', 'mannequin', 'background']
#   - index 2 (background) == REJECT the detection
#   - 224x224, ImageNet normalisation
# Used to prune YOLO false positives BEFORE geolocation/clustering.
# ----------------------------------------------------------------------------

CLASSIFIER_CLASSES = ["tent", "mannequin", "background"]


def load_classifier(clf_path, dev):
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T

    ckpt = torch.load(clf_path, map_location="cpu")
    num_out = 3

    def try_load(model_fn):
        m = model_fn(weights=None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_out)
        m.load_state_dict(ckpt)
        return m

    model = None
    for variant, fn in [("MobileNetV3-Large", tv_models.mobilenet_v3_large),
                        ("MobileNetV3-Small", tv_models.mobilenet_v3_small)]:
        try:
            model = try_load(fn)
            print(f"[classifier] loaded {clf_path} as {variant}", flush=True)
            break
        except RuntimeError:
            continue
    if model is None:
        raise RuntimeError("classifier checkpoint matched no MobileNetV3 variant")

    model.to(dev).eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform


def classify_crop(clf, transform, pil_img, box_xyxy, dev, margin=0.15):
    """Return (class_idx, confidence). class_idx 2 == background (reject).
    A small margin expands the crop so context helps the classifier."""
    import torch
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin; x2 += bw * margin
    y1 -= bh * margin; y2 += bh * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(pil_img.width, int(x2)), min(pil_img.height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    t = transform(crop).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = clf(t)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
    return idx, conf


# ----------------------------------------------------------------------------
# YOLO loading / inference
# ----------------------------------------------------------------------------

def load_model(model_path):
    from ultralytics import YOLO   # works for YOLO11 and RT-DETR weights
    model = YOLO(model_path)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    print(f"[model] loaded {model_path}  device={dev}", flush=True)
    return model, dev


def class_name_map(model):
    """Return {class_id: name}. We normalise to lowercase for matching."""
    names = model.names if hasattr(model, "names") else {}
    return {int(k): str(v).lower() for k, v in names.items()}


def run_inference(model, dev, image_path, conf):
    """Return list of detections: {cls_id, cls_name, conf, box_xyxy, cx, cy}."""
    res = model.predict(image_path, conf=conf, device=dev, verbose=False)[0]
    out = []
    names = class_name_map(model)
    if res.boxes is None:
        return out
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        cid = int(b.cls[0].item())
        out.append({
            "cls_id": cid,
            "cls_name": names.get(cid, str(cid)),
            "conf": float(b.conf[0].item()),
            "box_xyxy": [float(v) for v in xyxy],
            "cx": float((xyxy[0] + xyxy[2]) / 2.0),
            "cy": float((xyxy[1] + xyxy[3]) / 2.0),
        })
    return out


# ----------------------------------------------------------------------------
# Geolocation  (A) synthetic nadir back-projection
# ----------------------------------------------------------------------------

def backproject_nadir(cx_px, cy_px, cam_xyz, intr):
    """Pixel (cx,cy) -> ground (X,Y) for a pure-nadir camera at cam_xyz.

    Pinhole model. The camera looks straight down (-Z). A pixel offset from the
    image centre maps linearly to a ground offset scaled by altitude.

        ground_offset = (pixel_offset / focal_px) * altitude

    focal_px = focal_mm / sensor_mm * image_px   (per axis)

    Blender/most cameras: +x right, image +y DOWN. World here: +X right,
    +Y "up" in the top-down plot. We flip the Y pixel axis so the ground Y
    matches the world Y convention used by generate_flight.py.
    """
    W = intr["image_width_px"]; H = intr["image_height_px"]
    fmm = intr["focal_length_mm"]
    sw = intr["sensor_width_mm"]; sh = intr["sensor_height_mm"]
    fx = fmm / sw * W
    fy = fmm / sh * H
    alt = cam_xyz[2]

    dx_px = cx_px - W / 2.0
    dy_px = cy_px - H / 2.0
    gx = (dx_px / fx) * alt + cam_xyz[0]
    gy = (-dy_px / fy) * alt + cam_xyz[1]   # flip image-y to world-y
    return gx, gy


def backproject_raycast(cx_px, cy_px, cam_matrix_world, intr):
    """GIMBAL-LOCKED reprojection: yaw follows the drone, pitch/roll don't.

    Casts a ray from the camera centre through the detection pixel and
    intersects the ground plane z=0. On a nadir camera it reproduces
    backproject_nadir exactly (verified to ~1e-15 m).

    A real gimbal holds the camera level and pointing straight down
    regardless of the drone body's attitude -- only yaw (heading) tracks the
    drone. So rather than trusting whatever pitch/roll happens to be baked
    into the drone's pose matrix, we extract just the yaw from it and rebuild
    a nadir, zero-roll camera orientation from that yaw alone. This matches
    the model used for real flights in cluster_real_yaw.py's
    backproject_yaw_nadir(), just expressed as a rotation matrix instead of
    an explicit heading angle.

    cam_matrix_world : 4x4 Blender camera pose. Blender camera looks down its
    local -Z axis with +X right, +Y up. Returns (X, Y) ground coords, or None
    if the ray doesn't hit the ground in front of the camera.
    """
    M = np.asarray(cam_matrix_world, dtype=float)
    cam_pos = M[:3, 3]
    R_drone = M[:3, :3]

    # Recover yaw only, discarding drone pitch/roll: the camera's local +Y
    # ("up", image top) axis in world space still carries the yaw even under
    # drone pitch/roll, so its horizontal-plane projection gives the heading
    # the gimbal is pointed at.
    up_world = R_drone @ np.array([0.0, 1.0, 0.0])
    yaw = math.atan2(up_world[0], up_world[1])
    # Rebuild a nadir (local -Z looks straight down), zero-roll rotation from
    # yaw alone. Columns are where camera-local X/Y/Z (right/up/look-up) land
    # in world space; this is the fixed-pitch, yaw-only gimbal model.
    R = np.array([
        [math.cos(yaw), math.sin(yaw), 0.0],
        [-math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])

    W = intr["image_width_px"]; H = intr["image_height_px"]
    fmm = intr["focal_length_mm"]
    sw = intr["sensor_width_mm"]; sh = intr["sensor_height_mm"]
    fx = fmm / sw * W
    fy = fmm / sh * H
    # pixel -> camera-space ray dir (Blender cam: +x right, +y up, looks -z)
    x = (cx_px - W / 2.0) / fx
    y = -(cy_px - H / 2.0) / fy
    d_cam = np.array([x, y, -1.0])
    d_world = R @ d_cam
    n = np.linalg.norm(d_world)
    if n < 1e-12:
        return None
    d_world /= n
    if abs(d_world[2]) < 1e-9:
        return None
    t = -cam_pos[2] / d_world[2]
    if t < 0:
        return None
    hit = cam_pos + t * d_world
    return float(hit[0]), float(hit[1])


# ----------------------------------------------------------------------------
# Geolocation  (B) real GoPro Tier-1: detection inherits image GPS
# ----------------------------------------------------------------------------

def read_exif_gps(image_path):
    """Return (lat, lon, alt_m_or_None) in decimal degrees, or None if absent."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
    except Exception:
        print("[exif] Pillow not available", flush=True)
        return None
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        gps_ifd = None
        for tag, val in exif.items():
            if TAGS.get(tag) == "GPSInfo":
                gps_ifd = val
                break
        if not gps_ifd:
            return None
        gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

        def to_deg(dms, ref):
            d = float(dms[0]); m = float(dms[1]); s = float(dms[2])
            dec = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                dec = -dec
            return dec

        if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
            return None
        lat = to_deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = to_deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        alt = None
        if "GPSAltitude" in gps:
            try:
                alt = float(gps["GPSAltitude"])
            except Exception:
                alt = None
        return lat, lon, alt
    except Exception as e:
        print(f"[exif] {os.path.basename(image_path)}: {e}", flush=True)
        return None


def latlon_to_local_m(lat, lon, lat0, lon0):
    """Equirectangular projection to local metres about (lat0, lon0).
    Good enough over a competition-sized area."""
    R = 6378137.0
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R
    return x, y


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------

def dbscan_densest(points, eps, min_samples, confs=None, rank="confsum"):
    """Cluster ground points and pick the best cluster per the ranking mode.

    points   : Nx2 array of ground (X, Y)
    confs    : optional length-N array of detection confidences (0..1)
    rank     : how to score clusters and choose the winner:
        "count"        - most member points (original; loses to FP clumps)
        "confsum"      - highest SUM of confidences (DEFAULT; quality beats
                         quantity, eps-insensitive -- proven on real data)
        "confmean_count" - mean(conf) * count (balances both)

    Returns (labels, centroids_by_label, best_label, scores_by_label).
    Centroids are CONFIDENCE-WEIGHTED when confs are given (a high-confidence
    detection pulls the centroid toward it), else plain means.
    """
    from sklearn.cluster import DBSCAN
    if len(points) == 0:
        return np.array([]), {}, None, {}
    points = np.asarray(points, dtype=float)
    if confs is None:
        confs = np.ones(len(points))
    confs = np.asarray(confs, dtype=float)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    centroids, scores = {}, {}
    for lab in set(labels.tolist()):
        if lab == -1:
            continue
        mask = labels == lab
        w = confs[mask]
        pts = points[mask]
        wsum = w.sum()
        if wsum > 1e-9:
            cen = (pts * w[:, None]).sum(axis=0) / wsum
        else:
            cen = pts.mean(axis=0)
        centroids[int(lab)] = cen.tolist()
        if rank == "count":
            scores[int(lab)] = float(mask.sum())
        elif rank == "confmean_count":
            scores[int(lab)] = float(w.mean() * mask.sum())
        else:  # "confsum"
            scores[int(lab)] = float(w.sum())
    best = max(scores, key=scores.get) if scores else None

    # ROBUSTNESS FALLBACK: a sparsely-sampled target (e.g. tent seen in only
    # 2-3 frames) can be detected accurately yet rejected because it has fewer
    # than min_samples points. If nothing clustered but points DO exist, retry
    # with min_samples=1 so a small genuine cluster survives. The confidence-
    # weighted ranking then still rejects lone low-conf FPs vs a tight knot.
    if best is None and len(points) > 0 and min_samples > 1:
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(points)
        centroids, scores = {}, {}
        for lab in set(labels.tolist()):
            if lab == -1:
                continue
            mask = labels == lab
            w = confs[mask]; pts = points[mask]
            wsum = w.sum()
            cen = (pts * w[:, None]).sum(axis=0) / wsum if wsum > 1e-9 else pts.mean(axis=0)
            centroids[int(lab)] = cen.tolist()
            # in fallback, weight by BOTH count and confidence so a 3-point
            # tight knot beats scattered singletons
            scores[int(lab)] = float(w.sum())
        # require the winning fallback cluster to have at least 2 points OR
        # a single high-confidence detection, else it's probably just an FP
        viable = {l: s for l, s in scores.items()
                  if (labels == l).sum() >= 2 or s >= 0.5}
        best = max(viable, key=viable.get) if viable else None

    return labels, centroids, best, scores


# ----------------------------------------------------------------------------
# Pipeline (A): synthetic flight
# ----------------------------------------------------------------------------

def match_class(cls_name, target):
    """Loose match so 'mannequin'/'person'/'human' all count as mannequin etc."""
    cls_name = cls_name.lower()
    if target == "tent":
        return "tent" in cls_name
    if target == "mannequin":
        return any(k in cls_name for k in ("mannequin", "person", "human", "dummy"))
    return False


def run_synthetic(args):
    flight_dir = args.flight_dir
    log_path = os.path.join(flight_dir, "flight_log.jsonl")
    gt_path = os.path.join(flight_dir, "ground_truth.json")
    if not os.path.isfile(log_path):
        sys.exit(f"[fatal] no flight_log.jsonl in {flight_dir}")

    with open(log_path) as f:
        frames = [json.loads(l) for l in f if l.strip()]
    ground_truth = None
    if os.path.isfile(gt_path):
        with open(gt_path) as f:
            ground_truth = json.load(f)

    model, dev = load_model(args.model)

    # optional classifier wrapper to prune YOLO false positives
    clf = clf_tf = None
    if args.classifier:
        clf, clf_tf = load_classifier(args.classifier, dev)

    from PIL import Image

    per_frame = []
    pts = {"tent": [], "mannequin": []}          # accumulated ground points
    pts_meta = {"tent": [], "mannequin": []}      # parallel (frame, conf)
    clf_stats = {"kept": 0, "rejected": 0}

    used_raycast = False
    for fr in frames:
        img_path = os.path.join(flight_dir, fr["image"])
        dets = run_inference(model, dev, img_path, args.conf)
        cam_xyz = fr["cam_xyz_m"]
        intr = fr["intrinsics"]
        cam_M = fr.get("cam_matrix_world")   # present in all flight logs
        pil = Image.open(img_path).convert("RGB") if clf is not None else None
        frame_dets = []
        for d in dets:
            # classifier gate: reject if it says background (idx 2) or disagrees
            clf_idx = clf_conf = None
            if clf is not None:
                clf_idx, clf_conf = classify_crop(clf, clf_tf, pil, d["box_xyxy"], dev)
                d = {**d, "clf_idx": clf_idx, "clf_conf": clf_conf,
                     "clf_name": (CLASSIFIER_CLASSES[clf_idx] if clf_idx is not None else None)}
                # reject background, or low-confidence classifier calls
                if clf_idx is None or clf_idx == 2 or clf_conf < args.clf_conf:
                    clf_stats["rejected"] += 1
                    d["rejected"] = True
                    frame_dets.append({**d, "ground_xy": None})
                    continue
                clf_stats["kept"] += 1
            # GENERAL ray-cast reprojection (handles yaw/pitch/roll). Falls back
            # to the nadir formula only if the pose matrix is missing.
            if cam_M is not None and not args.force_nadir:
                hit = backproject_raycast(d["cx"], d["cy"], cam_M, intr)
                used_raycast = True
                if hit is None:
                    # ray missed the ground (steep tilt, edge pixel) -> skip
                    frame_dets.append({**d, "ground_xy": None})
                    continue
                gx, gy = hit
            else:
                gx, gy = backproject_nadir(d["cx"], d["cy"], cam_xyz, intr)
            rec = {**d, "ground_xy": [gx, gy]}
            frame_dets.append(rec)
            for tgt in ("tent", "mannequin"):
                # if classifier ran, trust ITS label; else trust YOLO's
                if clf is not None and clf_idx is not None and clf_idx < 2:
                    matched = (CLASSIFIER_CLASSES[clf_idx] == tgt)
                else:
                    matched = match_class(d["cls_name"], tgt)
                if matched:
                    pts[tgt].append([gx, gy])
                    pts_meta[tgt].append({"frame": fr["frame_index"], "conf": d["conf"]})
        per_frame.append({
            "frame_index": fr["frame_index"],
            "image": fr["image"],
            "cam_xyz_m": cam_xyz,
            "detections": frame_dets,
        })
        print(f"[frame {fr['frame_index']:04d}] {len(dets)} dets", flush=True)

    if clf is not None:
        print(f"[classifier] kept {clf_stats['kept']}, "
              f"rejected {clf_stats['rejected']} detections", flush=True)

    # cluster per class
    clusters_out = {}
    predictions = {}
    for tgt in ("tent", "mannequin"):
        arr = np.array(pts[tgt], dtype=float) if pts[tgt] else np.zeros((0, 2))
        confs = np.array([m["conf"] for m in pts_meta[tgt]], dtype=float) \
            if pts_meta[tgt] else np.zeros((0,))
        labels, centroids, best, scores = dbscan_densest(
            arr, args.eps, args.min_samples, confs=confs, rank=args.rank)
        clusters_out[tgt] = {
            "points": arr.tolist(),
            "labels": labels.tolist() if len(labels) else [],
            "centroids": centroids,
            "densest_label": best,
            "scores": scores,
            "rank_mode": args.rank,
            "point_meta": pts_meta[tgt],
        }
        if best is not None:
            predictions[tgt] = centroids[best]

    # error vs ground truth
    errors = {}
    if ground_truth and "ground_truth_world_xy" in ground_truth:
        gtxy = ground_truth["ground_truth_world_xy"]
        for tgt in ("tent", "mannequin"):
            if tgt in predictions and tgt in gtxy:
                px, py = predictions[tgt]
                tx, ty = gtxy[tgt]["x"], gtxy[tgt]["y"]
                err_m = math.hypot(px - tx, py - ty)
                errors[tgt] = {
                    "pred_xy": [px, py], "true_xy": [tx, ty],
                    "error_m": err_m,
                    "error_ft": err_m / 0.3048,
                    "within_50ft": (err_m / 0.3048) <= 50.0,
                }

    results = {
        "mode": "synthetic",
        "flight_dir": os.path.abspath(flight_dir),
        "params": {"conf": args.conf, "eps": args.eps, "min_samples": args.min_samples},
        "coord_system": "metres, world origin = ground centre",
        "ground_truth": ground_truth,
        "per_frame": per_frame,
        "clusters": clusters_out,
        "predictions": predictions,
        "errors": errors,
    }
    _write_and_report(results, args.out, errors)


# ----------------------------------------------------------------------------
# Pipeline (B): real GoPro images, Tier-1 GPS-per-image
# ----------------------------------------------------------------------------

def run_real(args):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(glob.glob(os.path.join(args.images_dir, ext)))
    files = sorted(set(files))
    if not files:
        sys.exit(f"[fatal] no images in {args.images_dir}")

    # establish local origin from the first image that has GPS
    origin = None
    gps_cache = {}
    for fpath in files:
        g = read_exif_gps(fpath)
        gps_cache[fpath] = g
        if g and origin is None:
            origin = (g[0], g[1])
    if origin is None:
        sys.exit("[fatal] no GPS EXIF found in any image -- was GoPro GPS enabled?")
    print(f"[exif] local origin lat0={origin[0]:.6f} lon0={origin[1]:.6f}", flush=True)

    model, dev = load_model(args.model)

    per_frame = []
    pts = {"tent": [], "mannequin": []}
    pts_meta = {"tent": [], "mannequin": []}

    for i, fpath in enumerate(files):
        g = gps_cache.get(fpath)
        dets = run_inference(model, dev, fpath, args.conf)
        if g is None:
            print(f"[frame {i:04d}] {os.path.basename(fpath)} NO GPS - skipped for geoloc", flush=True)
            lx = ly = None
        else:
            lx, ly = latlon_to_local_m(g[0], g[1], origin[0], origin[1])
        frame_dets = []
        for d in dets:
            # Tier-1: detection inherits the image's GPS-derived local coord.
            gxy = [lx, ly] if lx is not None else None
            rec = {**d, "ground_xy": gxy}
            frame_dets.append(rec)
            if gxy is not None:
                for tgt in ("tent", "mannequin"):
                    if match_class(d["cls_name"], tgt):
                        pts[tgt].append([lx, ly])
                        pts_meta[tgt].append({"frame": i, "conf": d["conf"]})
        per_frame.append({
            "frame_index": i,
            "image": os.path.basename(fpath),
            "gps": {"lat": g[0], "lon": g[1]} if g else None,
            "local_xy_m": [lx, ly] if lx is not None else None,
            "detections": frame_dets,
        })
        print(f"[frame {i:04d}] {os.path.basename(fpath)} {len(dets)} dets", flush=True)

    clusters_out = {}
    predictions = {}
    for tgt in ("tent", "mannequin"):
        arr = np.array(pts[tgt], dtype=float) if pts[tgt] else np.zeros((0, 2))
        confs = np.array([m["conf"] for m in pts_meta[tgt]], dtype=float) \
            if pts_meta[tgt] else np.zeros((0,))
        labels, centroids, best, scores = dbscan_densest(
            arr, args.eps_m, args.min_samples, confs=confs, rank=args.rank)
        clusters_out[tgt] = {
            "points": arr.tolist(),
            "labels": labels.tolist() if len(labels) else [],
            "centroids": centroids,
            "densest_label": best,
            "scores": scores,
            "rank_mode": args.rank,
            "point_meta": pts_meta[tgt],
        }
        if best is not None:
            predictions[tgt] = centroids[best]

    results = {
        "mode": "real",
        "images_dir": os.path.abspath(args.images_dir),
        "origin_latlon": origin,
        "params": {"conf": args.conf, "eps_m": args.eps_m, "min_samples": args.min_samples},
        "coord_system": "local metres about origin_latlon (equirectangular)",
        "per_frame": per_frame,
        "clusters": clusters_out,
        "predictions": predictions,
        "errors": {},
    }
    _write_and_report(results, args.out, {})


def _write_and_report(results, out_path, errors):
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\n[done] results -> {out_path}", flush=True)
    preds = results.get("predictions", {})
    for tgt in ("tent", "mannequin"):
        if tgt in preds:
            x, y = preds[tgt]
            line = f"  {tgt:<10} predicted ({x:+.2f}, {y:+.2f})"
            if tgt in errors:
                e = errors[tgt]
                ok = "OK" if e["within_50ft"] else "MISS"
                line += f"   error {e['error_m']:.2f} m / {e['error_ft']:.1f} ft  [{ok} vs 50ft]"
            print(line, flush=True)
        else:
            print(f"  {tgt:<10} NO CLUSTER FOUND (no detections clustered)", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("synthetic", help="run on a generate_flight.py output")
    a.add_argument("--flight_dir", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--conf", type=float, default=0.15)
    a.add_argument("--eps", type=float, default=2.0, help="DBSCAN eps in METRES")
    a.add_argument("--min_samples", type=int, default=3)
    a.add_argument("--rank", choices=("confsum", "count", "confmean_count"),
                   default="confsum",
                   help="cluster ranking: confsum (default, robust) beats raw count")
    a.add_argument("--classifier", default=None,
                   help="path to mobilenet_finetuned.pth; if set, each YOLO "
                        "detection is re-checked and background/low-conf ones "
                        "are dropped before clustering (prunes FPs)")
    a.add_argument("--clf_conf", type=float, default=0.5,
                   help="min classifier confidence to keep a detection (else reject)")
    a.add_argument("--force_nadir", action="store_true",
                   help="ignore the pose matrix's yaw and use the nadir formula "
                        "directly (only for debugging; default uses ray-cast, "
                        "which extracts yaw from the pose matrix but always "
                        "assumes gimbal-locked nadir pitch and zero roll)")
    a.add_argument("--out", default="results.json")
    a.set_defaults(func=run_synthetic)

    b = sub.add_parser("real", help="run on real geotagged GoPro images (Tier-1)")
    b.add_argument("--images_dir", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--conf", type=float, default=0.15)
    b.add_argument("--eps_m", type=float, default=5.0, help="DBSCAN eps in METRES")
    b.add_argument("--min_samples", type=int, default=2)
    b.add_argument("--rank", choices=("confsum", "count", "confmean_count"),
                   default="confsum",
                   help="cluster ranking: confsum (default, robust) beats raw count")
    b.add_argument("--out", default="results_real.json")
    b.set_defaults(func=run_real)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()