#!/usr/bin/env python3
"""
inspect_dataset.py -- what metadata do these drone images actually have?
========================================================================

Reports, per-image and in aggregate:
  - GPS lat/lon/alt
  - ANY orientation field (EXIF GPSImgDirection, XMP gimbal/flight yaw-pitch-roll)
  - camera intrinsics hints (make/model/focal/35mm-equiv/resolution)
  - flight path spread (did the drone actually move?)
  - a VERDICT: which reprojection tier is possible

Run:  python inspect_dataset.py watsa2
"""
import sys, os, glob, math, json, re

folder = sys.argv[1] if len(sys.argv) > 1 else "."
files = []
for ext in ("*.jpg","*.jpeg","*.JPG","*.JPEG","*.png","*.PNG","*.dng","*.DNG","*.tif","*.TIF"):
    files += glob.glob(os.path.join(folder, ext))
    files += glob.glob(os.path.join(folder, "**", ext), recursive=True)
files = sorted(set(files))
print(f"=== {len(files)} images in {folder} ===\n")
if not files:
    sys.exit(f"no images found -- check the folder path: {os.path.abspath(folder)}")

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def dms_to_deg(dms, ref):
    d, m, s = [float(x) for x in dms]
    v = d + m/60.0 + s/3600.0
    return -v if ref in ("S","W") else v


def get_exif(path):
    img = Image.open(path)
    exif = img._getexif() or {}
    return img, {TAGS.get(k,k): v for k,v in exif.items()}


def get_xmp_fields(path):
    """Scan raw bytes for XMP orientation/altitude fields (DJI, Autel, Skydio,
    Parrot all write these). Returns dict of found key->value."""
    raw = open(path, "rb").read()
    out = {}
    keys = [
        # DJI / generic drone XMP
        "GimbalYawDegree","GimbalPitchDegree","GimbalRollDegree",
        "FlightYawDegree","FlightPitchDegree","FlightRollDegree",
        "RelativeAltitude","AbsoluteAltitude",
        "GpsLatitude","GpsLongitude","GpsLatitude","LRFTargetDistance",
        # Autel / others
        "Yaw","Pitch","Roll","Heading",
        # Skydio
        "CameraOrientationNED",
    ]
    for k in keys:
        # match  key="value"  or  key>value<
        for pat in (rf'{k}\s*=\s*"([^"]*)"', rf'{k}>([^<]*)<'):
            m = re.search(pat.encode(), raw)
            if m:
                try:
                    out[k] = m.group(1).decode("latin-1").strip()
                except Exception:
                    pass
                break
    return out


# ---------------- sample image deep-dive ----------------
sample = files[0]
img, tags = get_exif(sample)
print(f"--- SAMPLE: {os.path.basename(sample)} ---")
print(f"  resolution      : {img.size[0]} x {img.size[1]}")
for key in ("Make","Model","LensModel","FocalLength","FocalLengthIn35mmFilm",
            "DigitalZoomRatio","ExifImageWidth","ExifImageHeight",
            "DateTimeOriginal","Orientation"):
    if key in tags:
        print(f"  {key:<16}: {tags[key]}")

gps = tags.get("GPSInfo")
sample_gps = None
if gps:
    g = {GPSTAGS.get(k,k): v for k,v in gps.items()}
    if "GPSLatitude" in g:
        lat = dms_to_deg(g["GPSLatitude"], g.get("GPSLatitudeRef","N"))
        lon = dms_to_deg(g["GPSLongitude"], g.get("GPSLongitudeRef","E"))
        sample_gps = (lat, lon)
        print(f"  GPS             : {lat:.7f}, {lon:.7f}")
    if "GPSAltitude" in g:
        print(f"  GPSAltitude     : {float(g['GPSAltitude']):.1f} m (above SEA LEVEL)")
    # EXIF heading fields -- these are what we need for yaw!
    for k in ("GPSImgDirection","GPSImgDirectionRef","GPSTrack","GPSTrackRef",
              "GPSDestBearing","GPSSpeed"):
        if k in g:
            print(f"  *** {k:<12}: {g[k]}   <-- HEADING DATA")

xmp = get_xmp_fields(sample)
print(f"\n  XMP / drone metadata:")
if xmp:
    for k, v in xmp.items():
        marker = "  <-- ORIENTATION" if any(t in k for t in ("Yaw","Pitch","Roll","Heading","Orientation")) else ""
        print(f"    {k:<22}: {v}{marker}")
else:
    print("    (none found)")

# ---------------- aggregate over all images ----------------
print(f"\n--- ALL {len(files)} IMAGES ---")
rows = []
n_gps = n_head_exif = n_head_xmp = 0
for f in files:
    rec = {"file": os.path.basename(f)}
    try:
        im, t = get_exif(f)
        rec["w"], rec["h"] = im.size
        gg = t.get("GPSInfo")
        if gg:
            d = {GPSTAGS.get(k,k): v for k,v in gg.items()}
            if "GPSLatitude" in d:
                rec["lat"] = dms_to_deg(d["GPSLatitude"], d.get("GPSLatitudeRef","N"))
                rec["lon"] = dms_to_deg(d["GPSLongitude"], d.get("GPSLongitudeRef","E"))
                n_gps += 1
            if "GPSAltitude" in d:
                rec["gps_alt"] = float(d["GPSAltitude"])
            if "GPSImgDirection" in d:
                rec["heading_exif"] = float(d["GPSImgDirection"])
                n_head_exif += 1
        x = get_xmp_fields(f)
        for k in ("GimbalYawDegree","FlightYawDegree","Yaw","Heading"):
            if k in x:
                try:
                    rec["heading_xmp"] = float(x[k]); n_head_xmp += 1
                except Exception:
                    pass
                break
        for k in ("GimbalPitchDegree","RelativeAltitude"):
            if k in x:
                rec[k] = x[k]
    except Exception as e:
        rec["error"] = str(e)
    rows.append(rec)

print(f"  with GPS            : {n_gps}/{len(files)}")
print(f"  with EXIF heading   : {n_head_exif}/{len(files)}")
print(f"  with XMP yaw        : {n_head_xmp}/{len(files)}")

coords = [(r["lat"], r["lon"]) for r in rows if "lat" in r]
if coords:
    lats = [c[0] for c in coords]; lons = [c[1] for c in coords]
    lat0 = sum(lats)/len(lats)
    span_ns = (max(lats)-min(lats))*111000
    span_ew = (max(lons)-min(lons))*111000*math.cos(math.radians(lat0))
    print(f"  flight spread       : {span_ns:.0f} m N-S  x  {span_ew:.0f} m E-W")
    print(f"  centre              : {lat0:.7f}, {sum(lons)/len(lons):.7f}")
alts = [r["gps_alt"] for r in rows if "gps_alt" in r]
if alts:
    print(f"  GPS alt range       : {min(alts):.1f} .. {max(alts):.1f} m (sea level)")
    print(f"    -> AGL = these minus ground elevation. ASK THE TEAM for AGL.")

# headings, if any
heads = [r.get("heading_exif", r.get("heading_xmp")) for r in rows]
heads = [h for h in heads if h is not None]
if heads:
    print(f"  heading range       : {min(heads):.0f} .. {max(heads):.0f} deg")
    print(f"    -> yaw VARIES by {max(heads)-min(heads):.0f} deg across the flight")

# per-image table
print(f"\n  {'file':<22} {'lat':>11} {'lon':>11} {'alt':>7} {'yaw':>7}")
for r in rows:
    print(f"  {r['file']:<22} {r.get('lat',float('nan')):>11.6f} "
          f"{r.get('lon',float('nan')):>11.6f} {r.get('gps_alt',float('nan')):>7.1f} "
          f"{r.get('heading_exif', r.get('heading_xmp', float('nan'))):>7.1f}")

# ---------------- verdict ----------------
print("\n=== VERDICT ===")
have_head = (n_head_exif > 0 or n_head_xmp > 0)
if have_head and n_gps == len(files):
    print("  FULL 6-DOF POSSIBLE: GPS + per-frame heading present.")
    print("  -> can do exact ray-cast reprojection (needs AGL + intrinsics).")
elif n_gps == len(files):
    print("  GPS ONLY, NO HEADING.")
    print("  -> nadir-assumption reprojection works, BUT if the drone yawed")
    print("     between shots the detections will smear (this is what happened")
    print("     with the last dataset).")
    print("  -> MITIGATION: heading can be ESTIMATED from consecutive GPS")
    print("     positions (bearing of travel). Works if the drone flew forward.")
else:
    print("  INSUFFICIENT GPS -- cannot geolocate.")

print("\n=== WHAT TO ASK THE TEAM ===")
print("  1. AGL (height above the track surface) -- REQUIRED for scale.")
print("  2. Was the camera locked straight down (nadir)? Any gimbal tilt?")
print("  3. Camera model + lens mode (linear vs wide) if not in EXIF above.")
print("  4. GPS pin of the actual tent and mannequin -> lets us measure error.")
print("  5. Did the drone fly forward along its heading (so GPS-bearing ~ yaw)?")

# dump machine-readable
out = os.path.join(folder, "_metadata_dump.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
print(f"\n[saved] per-image metadata -> {out}")