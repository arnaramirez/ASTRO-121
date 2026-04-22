"""
observe_hvc.py
--------------
Run this on the RPi / remote observing machine.

Uses:
- ugradio.leo     for Leuschner site coordinates
- ugradio.leusch  for telescope control
- astropy         for coordinate transforms and time handling

This script:
- builds a target list for the HVC project
- picks the nearest currently visible target
- points the dish
- acquires one block of data
- saves one .npz per pointing
- appends a CSV log
- saves remaining targets so the run can resume
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import ugradio
from ugradio import leusch

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u


# =========================================================
# Configuration
# =========================================================

PROJECT_NAME = "hvc"

# Project 12 region
L_MIN, L_MAX = 60.0, 180.0
B_MIN, B_MAX = 20.0, 60.0

# Start coarse for testing
DL = 4.0
DB = 4.0

# Number of spectra / integrations per pointing
NSPEC = 60

# Telescope safe bounds
ALT_MIN = 15.0
ALT_MAX = 85.0
AZ_MIN = 5.0
AZ_MAX = 350.0

# Waiting behavior
SLEEP_IF_NONE_VISIBLE = 300
RETRY_SLEEP = 10

# Output directory
UTC_DATE = Time.now().utc.strftime("%Y-%m-%d")
OUTDIR = Path(f"data/{UTC_DATE}").resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

LOGFILE = OUTDIR / "observation_log.csv"
STATEFILE = OUTDIR / "remaining_targets.json"

# Leuschner site from ugradio.leo
SITE = EarthLocation(
    lat=ugradio.leo.lat * u.deg,
    lon=ugradio.leo.lon * u.deg,
    height=ugradio.leo.alt * u.m,
)


# =========================================================
# Target helpers
# =========================================================

def make_serpentine_grid(l_min=L_MIN, l_max=L_MAX, b_min=B_MIN, b_max=B_MAX, dl=DL, db=DB):
    l_vals = np.arange(l_min, l_max + 0.1, dl, dtype=float)
    b_vals = np.arange(b_min, b_max + 0.1, db, dtype=float)

    targets = []
    for i, b in enumerate(b_vals):
        row = l_vals if i % 2 == 0 else l_vals[::-1]
        for l in row:
            targets.append((float(l), float(b)))
    return targets


def save_remaining_targets(remaining, path=STATEFILE):
    payload = [{"l_deg": l, "b_deg": b} for l, b in remaining]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_remaining_targets(path=STATEFILE):
    if not path.exists():
        return make_serpentine_grid()

    with open(path, "r") as f:
        payload = json.load(f)

    return [(float(row["l_deg"]), float(row["b_deg"])) for row in payload]


# =========================================================
# Coordinate helpers
# =========================================================

def gal_to_local(l_deg, b_deg, obstime):
    gal = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    icrs = gal.icrs
    altaz = gal.transform_to(AltAz(obstime=obstime, location=SITE))

    return {
        "ra_deg": icrs.ra.deg,
        "dec_deg": icrs.dec.deg,
        "alt_deg": altaz.alt.deg,
        "az_deg": altaz.az.deg,
    }


def observable(alt_deg, az_deg):
    return (ALT_MIN < alt_deg < ALT_MAX) and (AZ_MIN < az_deg < AZ_MAX)


def angular_sep_sq(alt1, az1, alt2, az2):
    da = abs(az1 - az2)
    da = min(da, 360.0 - da)
    de = alt1 - alt2
    return de**2 + da**2


def choose_next_target(remaining, tel):
    now = Time.now()

    try:
        cur_alt, cur_az = tel.get_pointing()
    except Exception:
        cur_alt, cur_az = 45.0, 180.0

    visible = []
    for idx, (l_deg, b_deg) in enumerate(remaining):
        info = gal_to_local(l_deg, b_deg, now)
        if observable(info["alt_deg"], info["az_deg"]):
            score = angular_sep_sq(cur_alt, cur_az, info["alt_deg"], info["az_deg"])
            visible.append((score, idx, l_deg, b_deg, info, now))

    if not visible:
        return None

    visible.sort(key=lambda x: x[0])
    return visible[0]


# =========================================================
# File / log helpers
# =========================================================

def make_file_stem(project, l_deg, b_deg, obstime):
    stamp = obstime.utc.strftime("%Y%m%dT%H%M%S")
    return f"{project}_l{int(round(l_deg)):03d}_b{int(round(b_deg)):03d}_{stamp}"


def append_log(row, logfile=LOGFILE):
    exists = logfile.exists()
    with open(logfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =========================================================
# Data acquisition / saving
# =========================================================

def get_integration_time_seconds():
    """
    Put the real spectrometer integration-time query here if available.
    If you do not have a direct query method, leave as NaN for now.
    """
    return np.nan


def acquire_pointing_data(l_deg, b_deg, nspec):
    """
    This is the ONE function you should adapt to the exact acquisition API
    available on the RPi.

    It should return a dictionary of numpy arrays, for example:
        {
            "spec":   array of shape (N, nchans),
            "times":  array of shape (N,),
            "alt_az": array of shape (N, 2),
        }

    Right now this is a template / placeholder because the exact spectrometer
    call available through ugradio on your RPi has not been shown here.
    """
    raise NotImplementedError(
        "Replace acquire_pointing_data() with the exact ugradio spectrometer "
        "call available on the RPi."
    )


def save_pointing_npz(npz_path, metadata, data_dict):
    """
    Explicitly save one pointing as an NPZ file.
    """
    npz_path = Path(npz_path).resolve()
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    save_dict.update(metadata)
    save_dict.update(data_dict)

    np.savez(npz_path, **save_dict)

    if not npz_path.exists():
        raise RuntimeError(f"NPZ file was not created: {npz_path}")
    if npz_path.stat().st_size == 0:
        raise RuntimeError(f"NPZ file is empty: {npz_path}")

    print(f"[save] Saved NPZ -> {npz_path} ({npz_path.stat().st_size} bytes)")
    return npz_path


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 70)
    print("Leuschner Project 12: High-Velocity Cloud Mapping")
    print("=" * 70)
    print(f"Site latitude : {ugradio.leo.lat:.4f} deg")
    print(f"Site longitude: {ugradio.leo.lon:.4f} deg")
    print(f"Site altitude : {ugradio.leo.alt:.1f} m")
    print(f"Output dir    : {OUTDIR}")
    print(f"Log file      : {LOGFILE}")
    print(f"State file    : {STATEFILE}")
    print()

    tel = leusch.LeuschTelescope()
    remaining = load_remaining_targets()
    save_remaining_targets(remaining)

    tint = get_integration_time_seconds()

    print(f"Targets remaining at start: {len(remaining)}")
    if np.isfinite(tint):
        print(f"Integration time per spectrum: {tint:.3f} s")
        print(f"Total time per pointing: {NSPEC * tint / 60:.2f} min")
    else:
        print("Integration time per spectrum: unknown")
    print()

    while remaining:
        chosen = choose_next_target(remaining, tel)

        if chosen is None:
            print(f"[sched] No visible targets. Sleeping {SLEEP_IF_NONE_VISIBLE} s.")
            time.sleep(SLEEP_IF_NONE_VISIBLE)
            continue

        _, idx, l_deg, b_deg, info, sched_time = chosen
        obs_time = Time.now()

        stem = make_file_stem(PROJECT_NAME, l_deg, b_deg, obs_time)
        npz_path = OUTDIR / f"{stem}.npz"

        print("-" * 70)
        print(f"Target: l={l_deg:.1f}, b={b_deg:.1f}")
        print(f"RA/Dec : {info['ra_deg']:.3f}, {info['dec_deg']:.3f} deg")
        print(f"Alt/Az : {info['alt_deg']:.3f}, {info['az_deg']:.3f} deg")
        print(f"UTC    : {obs_time.isot}")
        print(f"JD     : {obs_time.jd:.8f}")

        try:
            tel.point(info["alt_deg"], info["az_deg"], wait=True)
            alt_now, az_now = tel.get_pointing()

            # === actual acquisition ===
            data_dict = acquire_pointing_data(l_deg, b_deg, NSPEC)

            metadata = {
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": info["ra_deg"],
                "dec_deg": info["dec_deg"],
                "alt_deg_cmd": info["alt_deg"],
                "az_deg_cmd": info["az_deg"],
                "alt_deg_actual": alt_now,
                "az_deg_actual": az_now,
                "nspec": NSPEC,
                "int_time_each_s": float(tint) if np.isfinite(tint) else np.nan,
                "total_int_time_s": float(NSPEC * tint) if np.isfinite(tint) else np.nan,
            }

            npz_path = save_pointing_npz(npz_path, metadata, data_dict)

            append_log({
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "filename_npz": str(npz_path),
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": info["ra_deg"],
                "dec_deg": info["dec_deg"],
                "alt_deg_cmd": info["alt_deg"],
                "az_deg_cmd": info["az_deg"],
                "alt_deg_actual": alt_now,
                "az_deg_actual": az_now,
                "nspec": NSPEC,
                "int_time_each_s": float(tint) if np.isfinite(tint) else "",
                "total_int_time_s": float(NSPEC * tint) if np.isfinite(tint) else "",
                "npz_size_bytes": npz_path.stat().st_size,
                "status": "ok",
            })

            remaining.pop(idx)
            save_remaining_targets(remaining)

            print(f"[done] Saved {npz_path.name}")
            print(f"[done] Remaining targets: {len(remaining)}")

        except Exception as e:
            print(f"[error] Observation failed: {e}")

            append_log({
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "filename_npz": str(npz_path),
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": info["ra_deg"],
                "dec_deg": info["dec_deg"],
                "alt_deg_cmd": info["alt_deg"],
                "az_deg_cmd": info["az_deg"],
                "alt_deg_actual": "",
                "az_deg_actual": "",
                "nspec": NSPEC,
                "int_time_each_s": float(tint) if np.isfinite(tint) else "",
                "total_int_time_s": float(NSPEC * tint) if np.isfinite(tint) else "",
                "npz_size_bytes": "",
                "status": f"failed: {e}",
            })

            time.sleep(RETRY_SLEEP)

    print("\nAll targets completed.")


if __name__ == "__main__":
    main()
