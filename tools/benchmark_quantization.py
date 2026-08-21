#!/usr/bin/env python3
"""
benchmark_quantization.py -- how much does quantization cost us, end to end?
============================================================================

Runs the FULL localization pipeline (detect -> geolocate -> cluster -> rank)
once per model variant against the SAME rendered flight, and compares what
actually matters for the mission:

    * localization error per class, in metres, vs. known ground truth
    * whether each target still lands inside the 50 ft (15.24 m) scoring radius
    * detection yield per class (recall proxy feeding the cluster)
    * CONFIDENCE DISTRIBUTION SHIFT  <-- the quantization-specific risk
    * wall-clock inference time per frame

WHY CONFIDENCE SHIFT MATTERS HERE
Our cluster ranking scores clusters by SUMMED DETECTION CONFIDENCE, not by
point count. Quantization (especially INT8) compresses activation range and can
systematically flatten the confidence distribution. If true-target confidences
fall toward the false-positive band, the ranking margin erodes and a chance
concentration of FPs can win -- the exact 73.6 m failure mode we fixed by
switching to confidence weighting. Standard mAP will NOT reveal this. This
script reports the ranking margin explicitly.

Because the flight is pre-rendered, every model sees byte-identical frames, so
differences are attributable to the model alone.

--------------------------------------------------------------------------
STEP 1: export the variants (Ultralytics)
--------------------------------------------------------------------------
    from ultralytics import YOLO
    m = YOLO("yolo11m_best.pt")
    m.export(format="engine", half=True)                 # -> FP16 TensorRT
    m.export(format="engine", int8=True, data="suas.yaml")  # -> INT8 TensorRT
    m.export(format="onnx")                              # -> ONNX (portable)
    m.export(format="openvino", int8=True, data="suas.yaml")

NOTE: a TensorRT .engine is built for the SPECIFIC GPU it was exported on.
An engine built on the 4060 will NOT load on the Jetson Orin Nano. Export
on the Jetson for deployment numbers. Run this script on the laptop for the
ACCURACY comparison (valid anywhere), and re-run on the Jetson for LATENCY.

--------------------------------------------------------------------------
STEP 2: benchmark
--------------------------------------------------------------------------
    python benchmark_quantization.py \
        --flight_dir flight_yaw_only \
        --models yolo11m_best.pt yolo11m_best.engine yolo11m_best_int8.engine \
        --labels FP32 FP16-TRT INT8-TRT \
        --classifier mobilenet_finetuned.pth \
        --conf 0.15 --eps 3.0 --min_samples 3 \
        --out_dir quant_bench
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

FT_PER_M = 1.0 / 0.3048
RADIUS_M = 15.24          # 50 ft scoring radius


def run_pipeline(pipeline, flight_dir, model, out_json, args):
    """Shell out to the validated cluster_pipeline.py so we benchmark exactly
    the code that flies, not a reimplementation."""
    cmd = [sys.executable, pipeline, "synthetic",
           "--flight_dir", flight_dir,
           "--model", model,
           "--conf", str(args.conf),
           "--eps", str(args.eps),
           "--min_samples", str(args.min_samples),
           "--rank", args.rank,
           "--out", out_json]
    if args.classifier:
        cmd += ["--classifier", args.classifier, "--clf_conf", str(args.clf_conf)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return None, wall
    return proc.stdout, wall


def summarize(results_path, wall, n_frames_hint=None):
    """Pull the mission-relevant numbers out of a results.json."""
    with open(results_path) as f:
        r = json.load(f)
    out = {"per_class": {}}
    n_frames = len(r.get("per_frame", []))
    out["n_frames"] = n_frames
    out["wall_s"] = wall
    out["ms_per_frame"] = (wall / n_frames * 1000.0) if n_frames else float("nan")

    errs = r.get("errors", {}) or {}
    for tgt in ("tent", "mannequin"):
        cl = (r.get("clusters", {}) or {}).get(tgt, {}) or {}
        pts = cl.get("points", []) or []
        meta = cl.get("point_meta", []) or []
        confs = [m.get("conf", 0.0) for m in meta]
        scores = cl.get("scores", {}) or {}
        # ranking margin: winning cluster score vs. best runner-up.
        # A small margin means quantization could flip the decision.
        sv = sorted([float(v) for v in scores.values()], reverse=True)
        margin = (sv[0] - sv[1]) if len(sv) >= 2 else (sv[0] if sv else 0.0)
        rel_margin = (margin / sv[0]) if sv and sv[0] > 0 else float("nan")

        e = errs.get(tgt)
        out["per_class"][tgt] = {
            "n_points": len(pts),
            "n_clusters": len(scores),
            "mean_conf": (sum(confs) / len(confs)) if confs else float("nan"),
            "max_conf": max(confs) if confs else float("nan"),
            "win_score": sv[0] if sv else float("nan"),
            "margin_abs": margin,
            "margin_rel": rel_margin,
            "error_m": e["error_m"] if e else None,
            "within_50ft": e["within_50ft"] if e else None,
            "predicted": r.get("predictions", {}).get(tgt),
        }
    return out


def fmt(v, nd=2, dash="--"):
    if v is None:
        return dash
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return dash
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight_dir", required=True,
                    help="a rendered synthetic flight WITH ground_truth.json")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model files to compare (.pt/.engine/.onnx/openvino dir)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="display names, same order as --models")
    ap.add_argument("--pipeline", default="cluster_pipeline.py")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--min_samples", type=int, default=3)
    ap.add_argument("--rank", default="confsum",
                    choices=("confsum", "count", "confmean_count"))
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--clf_conf", type=float, default=0.5)
    ap.add_argument("--out_dir", default="quant_bench")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.flight_dir, "ground_truth.json")):
        sys.exit(f"[fatal] {args.flight_dir} has no ground_truth.json -- "
                 f"error cannot be measured without it")
    if not os.path.isfile(args.pipeline):
        sys.exit(f"[fatal] pipeline not found: {args.pipeline}")

    labels = args.labels if args.labels and len(args.labels) == len(args.models) \
        else [os.path.basename(m) for m in args.models]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 74)
    print(f"  flight     : {args.flight_dir}")
    print(f"  variants   : {len(args.models)}")
    print(f"  ranking    : {args.rank}   conf={args.conf} eps={args.eps} "
          f"min_samples={args.min_samples}")
    print(f"  classifier : {args.classifier or 'off'}")
    print("=" * 74 + "\n")

    summaries = {}
    for model, label in zip(args.models, labels):
        if not os.path.exists(model):
            print(f"[skip] {label}: not found ({model})\n")
            continue
        print(f"[run ] {label}  <- {model}", flush=True)
        out_json = os.path.join(args.out_dir, f"results_{label.replace('/','_')}.json")
        stdout, wall = run_pipeline(args.pipeline, args.flight_dir, model,
                                    out_json, args)
        if stdout is None:
            print(f"[fail] {label}: pipeline error (see above)\n")
            continue
        summaries[label] = summarize(out_json, wall)
        s = summaries[label]
        print(f"       {s['n_frames']} frames, {wall:.1f}s "
              f"({s['ms_per_frame']:.0f} ms/frame)\n", flush=True)

    if not summaries:
        sys.exit("[fatal] no variant completed successfully")

    # ---------------- comparison tables ----------------
    base_label = labels[0] if labels[0] in summaries else list(summaries)[0]

    for tgt in ("tent", "mannequin"):
        print("=" * 74)
        print(f"  {tgt.upper()}")
        print("=" * 74)
        hdr = (f"  {'variant':<14}{'err(m)':>9}{'err(ft)':>9}{'50ft':>7}"
               f"{'pts':>6}{'meanC':>8}{'margin%':>9}")
        print(hdr)
        print("  " + "-" * 70)
        for label, s in summaries.items():
            c = s["per_class"][tgt]
            err = c["error_m"]
            ok = ("PASS" if c["within_50ft"] else "MISS") if c["within_50ft"] is not None else "--"
            relm = c["margin_rel"] * 100 if c["margin_rel"] == c["margin_rel"] else float("nan")
            print(f"  {label:<14}{fmt(err):>9}"
                  f"{fmt(err*FT_PER_M if err is not None else None,1):>9}"
                  f"{ok:>7}{c['n_points']:>6}{fmt(c['mean_conf'],3):>8}"
                  f"{fmt(relm,1):>9}")
        # deltas vs baseline
        b = summaries[base_label]["per_class"][tgt]
        if b["error_m"] is not None:
            print("  " + "-" * 70)
            print(f"  delta vs {base_label}:")
            for label, s in summaries.items():
                if label == base_label:
                    continue
                c = s["per_class"][tgt]
                if c["error_m"] is None:
                    print(f"    {label:<12} NO CLUSTER (localization lost)")
                    continue
                d_err = c["error_m"] - b["error_m"]
                d_pts = c["n_points"] - b["n_points"]
                d_cnf = (c["mean_conf"] - b["mean_conf"]) if b["mean_conf"] == b["mean_conf"] else float("nan")
                print(f"    {label:<12} err {d_err:+.2f} m   points {d_pts:+d}   "
                      f"mean conf {fmt(d_cnf,3,'--')}")
        print()

    # ---------------- latency ----------------
    print("=" * 74)
    print("  THROUGHPUT  (this machine only -- re-run on the Jetson to deploy)")
    print("=" * 74)
    print(f"  {'variant':<14}{'total(s)':>11}{'ms/frame':>11}{'speedup':>10}")
    print("  " + "-" * 70)
    b_ms = summaries[base_label]["ms_per_frame"]
    for label, s in summaries.items():
        sp = (b_ms / s["ms_per_frame"]) if s["ms_per_frame"] else float("nan")
        print(f"  {label:<14}{s['wall_s']:>11.1f}{s['ms_per_frame']:>11.0f}"
              f"{fmt(sp,2):>10}x")

    # ---------------- verdict ----------------
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    for label, s in summaries.items():
        bad = []
        for tgt in ("tent", "mannequin"):
            c = s["per_class"][tgt]
            if c["error_m"] is None:
                bad.append(f"{tgt}: NO CLUSTER")
            elif not c["within_50ft"]:
                bad.append(f"{tgt}: {c['error_m']:.1f} m OUTSIDE 50 ft")
            elif c["margin_rel"] == c["margin_rel"] and c["margin_rel"] < 0.20:
                bad.append(f"{tgt}: ranking margin only "
                           f"{c['margin_rel']*100:.0f}% (fragile)")
        status = "OK -- both targets localized inside 50 ft" if not bad else "; ".join(bad)
        print(f"  {label:<14} {status}")

    print("\n  Reading the numbers:")
    print("   * err(m) is the only mission metric -- under 15.24 m scores.")
    print("   * points  = detections feeding the cluster. A big drop means")
    print("     quantization cost recall; clustering can absorb some of this.")
    print("   * meanC   = mean detection confidence. INT8 often flattens this.")
    print("   * margin% = how far the winning cluster beat the runner-up.")
    print("     Under ~20% the ranking is fragile: a small confidence shift")
    print("     could select a false-positive cluster instead. Watch this even")
    print("     when the error looks fine.")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[saved] {os.path.join(args.out_dir, 'summary.json')}")
    print(f"[saved] per-variant results_*.json in {args.out_dir}/ "
          f"(loadable in viewer.html)")


if __name__ == "__main__":
    main()
