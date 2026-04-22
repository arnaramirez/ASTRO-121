"""
sync_hvc_watcher.py
-------------------
Mirror the current Project 12 Leuschner observing directory from the Raspberry Pi
to your local machine using rsync over a jump host.

This script is designed for the observe_hvc.py layout:
    data/YYYY-MM-DD/hvc_lab4_YYYYMMDD/
        *.fits
        *.json
        manifest_latest.npz
        session_summary.json

Unlike the old Sun watcher, this does NOT delete remote files.
It is meant to create a local backup during observing.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import json
import os

# Optional: verify FITS files if astropy is available
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUMP_HOST = "radiolab@leuschner.berkeley.edu"   # or your ssh config alias jump host
RPI_USER  = "pi"
RPI_HOST  = "192.168.1.154"

# Must match the observing script's root on the Pi
REMOTE_BASE_DIR = "/home/pi/data"

# Session naming from observe_hvc.py
PROJECT_TAG = "hvc_lab4"

today = datetime.now().strftime("%Y-%m-%d")
session_name = f"{PROJECT_TAG}_{datetime.now().strftime('%Y%m%d')}"

REMOTE_SESSION_DIR = f"{REMOTE_BASE_DIR}/{today}/{session_name}"
LOCAL_BASE_DIR = Path(f"data/{today}")
LOCAL_SESSION_DIR = LOCAL_BASE_DIR / session_name
LOCAL_SESSION_DIR.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL_SEC = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def remote_session_exists():
    cmd = [
        "ssh",
        "-J", JUMP_HOST,
        f"{RPI_USER}@{RPI_HOST}",
        f"test -d '{REMOTE_SESSION_DIR}'"
    ]
    result = run_cmd(cmd)
    return result.returncode == 0

def list_remote_files():
    """
    List files in the remote session directory that we care about.
    """
    remote_cmd = (
        f"find '{REMOTE_SESSION_DIR}' -maxdepth 1 -type f "
        f"\\( -name '*.fits' -o -name '*.json' -o -name '*.npz' \\) | sort"
    )
    cmd = [
        "ssh",
        "-J", JUMP_HOST,
        f"{RPI_USER}@{RPI_HOST}",
        f"bash -lc \"{remote_cmd}\""
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        print("[sync] SSH/list error:", result.stderr.strip())
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def rsync_session():
    """
    Mirror the remote session directory to the local machine.
    """
    remote = f"{RPI_USER}@{RPI_HOST}:{REMOTE_SESSION_DIR}/"
    cmd = [
        "rsync",
        "-avz",
        "--partial",
        "--append-verify",
        "--ignore-existing",
        "-e", f"ssh -J {JUMP_HOST}",
        remote,
        str(LOCAL_SESSION_DIR) + "/",
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        print("[sync] rsync error:")
        print(result.stderr.strip())
        return False
    if result.stdout.strip():
        print(result.stdout.strip())
    return True

def rsync_manifest_and_summary():
    """
    Re-pull files that may legitimately change over time.
    Since manifest_latest.npz and session_summary.json are rewritten by the observer,
    we should sync them every cycle without --ignore-existing behavior.
    """
    for filename in ("manifest_latest.npz", "session_summary.json"):
        remote_file = f"{REMOTE_SESSION_DIR}/{filename}"
        local_file = LOCAL_SESSION_DIR / filename

        cmd = [
            "rsync",
            "-avz",
            "--partial",
            "--append-verify",
            "-e", f"ssh -J {JUMP_HOST}",
            f"{RPI_USER}@{RPI_HOST}:{remote_file}",
            str(local_file),
        ]
        result = run_cmd(cmd)
        # It's okay if these files do not exist yet
        if result.returncode != 0 and "No such file or directory" not in result.stderr:
            print(f"[sync] Warning: failed to sync {filename}")
            print(result.stderr.strip())

def verify_json(path):
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except Exception as e:
        print(f"[verify] JSON failed for {path.name}: {e}")
        return False

def verify_fits(path):
    if not HAS_ASTROPY:
        # Fallback: file exists and is non-empty
        return path.exists() and path.stat().st_size > 0

    try:
        with fits.open(path) as hdul:
            # Just touching the headers/data is enough for a basic integrity check
            _ = len(hdul)
            _ = hdul[0].header
        return True
    except Exception as e:
        print(f"[verify] FITS failed for {path.name}: {e}")
        return False

def verify_local_session():
    """
    Basic integrity checks on newly synced files.
    """
    ok = True

    for path in LOCAL_SESSION_DIR.glob("*.json"):
        if not verify_json(path):
            ok = False

    for path in LOCAL_SESSION_DIR.glob("*.fits"):
        if not verify_fits(path):
            ok = False

    manifest = LOCAL_SESSION_DIR / "manifest_latest.npz"
    if manifest.exists():
        try:
            import numpy as np
            d = np.load(manifest, allow_pickle=True)
            required = {"id", "l_deg", "b_deg", "jd", "alt_deg", "az_deg", "fits_file", "used_noise_cal"}
            if not required.issubset(d.files):
                print(f"[verify] Manifest missing keys: {required - set(d.files)}")
                ok = False
        except Exception as e:
            print(f"[verify] Manifest failed: {e}")
            ok = False

    return ok

def print_status():
    files = list(LOCAL_SESSION_DIR.glob("*"))
    n_fits = len(list(LOCAL_SESSION_DIR.glob("*.fits")))
    n_json = len(list(LOCAL_SESSION_DIR.glob("*.json")))
    has_manifest = (LOCAL_SESSION_DIR / "manifest_latest.npz").exists()
    has_summary = (LOCAL_SESSION_DIR / "session_summary.json").exists()

    print(f"[sync] Local dir: {LOCAL_SESSION_DIR.resolve()}")
    print(f"[sync] Files total: {len(files)} | FITS: {n_fits} | JSON: {n_json} | "
          f"manifest: {has_manifest} | summary: {has_summary}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("[sync] Starting HVC sync watcher")
    print(f"[sync] Remote session: {RPI_USER}@{RPI_HOST}:{REMOTE_SESSION_DIR}")
    print(f"[sync] Local session : {LOCAL_SESSION_DIR.resolve()}")
    print(f"[sync] Interval      : {SYNC_INTERVAL_SEC} s")
    print()

    while True:
        if not remote_session_exists():
            print("[sync] Remote session directory not found yet. Waiting...")
            time.sleep(SYNC_INTERVAL_SEC)
            continue

        remote_files = list_remote_files()
        print(f"[sync] Remote files currently visible: {len(remote_files)}")

        ok1 = rsync_session()
        rsync_manifest_and_summary()

        if ok1:
            verified = verify_local_session()
            if verified:
                print("[sync] Local mirror verified.")
            else:
                print("[sync] Local mirror has some files that did not verify cleanly.")

        print_status()
        print("-" * 72)
        time.sleep(SYNC_INTERVAL_SEC)


if __name__ == "__main__":
    main()
