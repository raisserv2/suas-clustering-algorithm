#!/usr/bin/env python3
"""
feed_flight.py -- replay an image folder the way a drone streams it
==================================================================

Sends images ONE AT A TIME, in filename order, with a fixed gap, so you can
exercise src/streaming_localizer.py end to end -- including across two
machines.

SRC and DST may each be a local path or  user@host:/path  (anything ssh /
rsync understand). Typical setups:

    # run ON the machine that holds the images ("drone"), push to the
    # machine running the pipeline ("companion"):
    python tools/feed_flight.py ./flight_images companion@192.168.1.50:~/incoming -i 2

    # run ON the pipeline machine, pull from the drone:
    python tools/feed_flight.py drone@192.168.1.60:/data/flight1 ./incoming -i 2

    # single-machine smoke test (two terminals):
    python tools/feed_flight.py ./flight_images ./incoming -i 2

Each image is delivered with rsync, which copies to a hidden temp name and
atomically renames it into place, so the watcher never sees a half-written
file (its --stable_sec check is a second guard).

If an OpenDroneMap geo.txt (pose per image) sits beside SRC or in its parent,
it is sent once before the first image so the watcher has pose from frame 0.
Override its location with --geo, disable with --no-geo.

ORDER OF OPERATIONS
    1. Start src/streaming_localizer.py on the DST side first, pointed at the
       incoming dir while it is still EMPTY (it ignores images already there
       unless you pass --process_existing).
    2. Then start this feeder.

Requires: rsync + ssh on both ends (already present on Jetson / Raspberry Pi /
any Linux). This script itself has no non-stdlib dependencies.
"""

import argparse
import os
import posixpath
import shlex
import subprocess
import sys
import time

IMG_EXTS = (".jpg", ".jpeg", ".png")


def is_remote(p):
    """True for  user@host:/path  / host:path, False for local paths
    (including Windows  C:\\...)."""
    if "://" in p or os.path.exists(p):
        return False
    if len(p) >= 3 and p[1] == ":" and p[2] in "\\/":
        return False                      # Windows drive letter
    return ":" in p


def list_images(src):
    """Return image paths under src, filename-sorted. Each entry is in the
    same local-or-remote form as src."""
    if is_remote(src):
        host, path = src.split(":", 1)
        finder = (
            "find " + shlex.quote(path) + " -maxdepth 4 -type f "
            r"\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print"
        )
        res = subprocess.run(["ssh", host, finder],
                             capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"[feed] listing {src} failed:\n{res.stderr.strip()}")
        files = sorted(l for l in res.stdout.splitlines() if l.strip())
        return [f"{host}:{f}" for f in files]

    files = []
    for root, _, names in os.walk(src):
        for n in names:
            if n.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, n))
    return sorted(files)


def join_path(base, name):
    """base + name, keeping the local-or-remote form of base."""
    if is_remote(base):
        host, path = base.split(":", 1)
        return f"{host}:{posixpath.join(path, name)}"
    return os.path.join(base, name)


def parent_path(p):
    if is_remote(p):
        host, path = p.split(":", 1)
        return f"{host}:{posixpath.dirname(path.rstrip('/'))}"
    return os.path.dirname(os.path.abspath(p.rstrip("/")))


def base_name(p):
    if is_remote(p):
        p = p.split(":", 1)[1]
    return posixpath.basename(p.replace("\\", "/"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="image folder (local or user@host:/path)")
    ap.add_argument("dst", help="incoming folder the watcher reads "
                                "(local or user@host:/path)")
    ap.add_argument("-i", "--interval", type=float, default=2.0,
                    help="seconds between images (default 2)")
    ap.add_argument("-n", "--limit", type=int, default=0,
                    help="send at most N images (0 = all)")
    ap.add_argument("--start-delay", type=float, default=0.0,
                    help="wait this long before the first image")
    ap.add_argument("--geo", default=None,
                    help="path to a geo.txt to deliver once before the images "
                         "(default: geo.txt beside SRC, or in its parent)")
    ap.add_argument("--no-geo", action="store_true",
                    help="do not send any geo.txt")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the send order and exit")
    args = ap.parse_args()

    files = list_images(args.src)
    if not files:
        sys.exit(f"[feed] no images under {args.src}")
    if args.limit:
        files = files[:args.limit]

    print(f"[feed] {len(files)} images   {args.src} -> {args.dst}   "
          f"every {args.interval}s", flush=True)
    if args.dry_run:
        for f in files:
            print("  " + base_name(f))
        return

    dst = args.dst.rstrip("/") + "/"
    if not is_remote(args.dst):
        os.makedirs(args.dst, exist_ok=True)

    # deliver geo.txt first so the watcher has pose from frame 0
    if not args.no_geo:
        geo = args.geo
        if geo is None:
            for cand in (join_path(args.src, "geo.txt"),
                         join_path(parent_path(args.src), "geo.txt")):
                if is_remote(cand):
                    geo = cand          # can't stat remote; let rsync try
                    break
                if os.path.isfile(cand):
                    geo = cand
                    break
        if geo:
            rc = subprocess.run(["rsync", "-q", geo, dst]).returncode
            print(f"[feed] geo.txt {'-> ' + args.dst if rc == 0 else f'skipped (rsync rc={rc})'}",
                  flush=True)
        else:
            print("[feed] no geo.txt found beside SRC -- watcher will use EXIF",
                  flush=True)

    if args.start_delay:
        time.sleep(args.start_delay)

    t0 = time.time()
    sent = 0
    for k, f in enumerate(files, 1):
        rc = subprocess.run(["rsync", "-q", f, dst]).returncode
        sent += (rc == 0)
        print(f"[feed {k:04d}/{len(files)}] {base_name(f)}   "
              f"{'ok' if rc == 0 else f'rsync rc={rc}'}", flush=True)
        if k < len(files):
            # hold a steady cadence even if a transfer took a moment
            time.sleep(max(0.0, t0 + k * args.interval - time.time()))

    print(f"[feed] done -- {sent}/{len(files)} delivered in "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
