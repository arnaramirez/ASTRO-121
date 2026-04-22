"""
observe_hvc.py
--------------
Leuschner observing script for Lab 4 / Project 12:
"Mapping a Big High-Velocity Cloud"

Designed to:
- use ugradio.leusch.LeuschTelescope for pointing
- use ugradio.leusch.Spectrometer for saved spectra
- observe in Galactic coordinates
- minimize slew time by greedily choosing the next visible target
- avoid expensive azimuth wrap crossings near 0/360 deg
- save data immediately to disk for every pointing

Outputs:
    data/YYYY-MM-DD/hvc_lab4_YYYYMMDD/
        point_XXXX_lLLL.L_bBB.B_<timestamp>.fits
        point_XXXX_lLLL.L_bBB.B_<timestamp>.json
        manifest_latest.npz
        session_summary.json
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ugradio
from ugradio import leo, leusch

from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u


# ============================================================================
# User configuration
# ============================================================================

# Project 12 bounds from the lab manual:
# roughly (l, b) = (60 -> 180 deg, 20 -> 60 deg), weak signal, few min/point
L_MIN = 60.0
L_MAX = 180.0
B_MIN = 20.0
B_MAX = 60.0

# Angular sampling:
# lab recommends ~2 deg in latitude and ~2/cos(b) in longitude to avoid
# oversampling in l away from the plane.
DB_DEG = 2.0
DL_EQUIV_DEG = 2.0

# Telescope safe limits from leusch.py / lab manual
ALT_MIN = 15.5   # use a small margin above the hard limit
ALT_MAX = 84.5   # use a small margin below the hard limit
AZ_MIN = 5.0
AZ_MAX = 350.0

# Integration settings
INTEGRATION_MIN = 3.0         # "at least a few minutes per point"
CAL_EVERY_N_POINTS = 8        # noise diode calibration cadence
CAL_NSPEC = 1                 # per calibration capture

# Scheduling / cadence
SETTLE_SEC = 3.0              # small pause after a slew
RECHECK_VISIBLE_SEC = 60.0    # if nothing visible, wait and re-check

# Output naming
PROJECT_TAG = "hvc_lab4"

# Spectrometer system keyword:
# leusch.Spectrometer.read_spec(..., coords, system='ga')
COORD_SYSTEM = "ga"


# ============================================================================
# Astropy site object using ugradio.leo
# ============================================================================

SITE = EarthLocation(
    lat=leo.lat * u.deg,
    lon=leo.lon * u.deg,
    height=leo.alt * u.m
)


# ============================================================================
# Helpers
# ============================================================================

def now_utc():
    return datetime.now(timezone.utc)

def timestamp():
    return now_utc().strftime("%Y%m%d_%H%M%S")

def jd_now():
    return Time(now_utc()).jd

def gal_to_altaz(l_deg, b_deg, obstime=None):
    """Convert Galactic (l,b) in deg to apparent Alt/Az in deg."""
    if obstime is None:
        obstime = Time(now_utc())

    gal = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    altaz = gal.transform_to(AltAz(obstime=obstime, location=SITE))
    return float(altaz.alt.deg), float(altaz.az.deg)

def angular_sep_deg(alt1, az1, alt2, az2):
    """
    Approximate slew distance cost.
    Uses wrapped azimuth difference but separately penalizes wrap crossings.
    """
    da = abs(alt2 - alt1)
    dz = wrapped_az_sep_deg(az1, az2)
    return np.hypot(da, dz)

def wrapped_az_sep_deg(az1, az2):
    """Shortest azimuth separation in degrees."""
    d = abs(az2 - az1) % 360.0
    return min(d, 360.0 - d)

def crosses_az_wrap(az1, az2, wrap_margin=12.0):
    """
    Return True if moving between az1 and az2 would likely force
    an unpleasant 0/360 boundary crossing in practice.
    """
    # This is conservative on purpose.
    low1 = az1 < wrap_margin
    high1 = az1 > (360.0 - wrap_margin)
    low2 = az2 < wrap_margin
    high2 = az2 > (360.0 - wrap_margin)
    return (low1 and high2) or (high1 and low2)

def is_pointable(alt_deg, az_deg):
    """Check dish limits with a small safety margin."""
    return (ALT_MIN <= alt_deg <= ALT_MAX) and (AZ_MIN <= az_deg <= AZ_MAX)

def build_hvc_grid():
    """
    Build a Galactic grid covering the Project 12 region.
    Uses db = 2 deg and dl = 2/cos(b) so true angular spacing stays ~2 deg.
    """
    targets = []
    point_id = 0

    b_vals = np.arange(B_MIN, B_MAX + 0.001, DB_DEG)
    for ib, b in enumerate(b_vals):
        cosb = np.cos(np.deg2rad(b))
        dl = DL_EQUIV_DEG / max(cosb, 1e-6)

        l_vals = np.arange(L_MIN, L_MAX + 0.001, dl)

        # snake pattern by b row; this helps, but final selection is still greedy
        if ib % 2 == 1:
            l_vals = l_vals[::-1]

        for l in l_vals:
            targets.append({
                "id": point_id,
                "l_deg": float(l),
                "b_deg": float(b),
                "observed": False,
            })
            point_id += 1

    return targets

def choose_next_target(targets, cur_alt, cur_az):
    """
    Greedy scheduler:
    among currently visible targets, choose the one with the smallest slew cost,
    plus a strong penalty for azimuth wrap crossing.
    """
    obstime = Time(now_utc())
    candidates = []

    for tgt in targets:
        if tgt["observed"]:
            continue

        alt, az = gal_to_altaz(tgt["l_deg"], tgt["b_deg"], obstime=obstime)
        if not is_pointable(alt, az):
            continue

        cost = angular_sep_deg(cur_alt, cur_az, alt, az)

        # Penalize wrap crossings heavily to avoid 6 deg -> 354 deg disasters
        if crosses_az_wrap(cur_az, az):
            cost += 200.0

        # Slight preference for higher altitude observations
        # because they are generally cleaner and remain visible longer
        cost += 0.03 * (85.0 - alt)

        candidates.append((cost, alt, az, tgt))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, alt, az, tgt = candidates[0]
    return tgt, alt, az

def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def update_manifest_npz(manifest_path, records):
    """
    Save a lightweight running manifest that is always rewritten atomically.
    """
    tmp = str(manifest_path) + ".tmp"
    ids = np.array([r["id"] for r in records], dtype=int)
    l_deg = np.array([r["l_deg"] for r in records], dtype=float)
    b_deg = np.array([r["b_deg"] for r in records], dtype=float)
    jd = np.array([r["jd"] for r in records], dtype=float)
    alt = np.array([r["alt_deg"] for r in records], dtype=float)
    az = np.array([r["az_deg"] for r in records], dtype=float)
    files = np.array([r["fits_file"] for r in records], dtype=object)
    cal = np.array([r["used_noise_cal"] for r in records], dtype=bool)

    np.savez(
        tmp,
        id=ids,
        l_deg=l_deg,
        b_deg=b_deg,
        jd=jd,
        alt_deg=alt,
        az_deg=az,
        fits_file=files,
        used_noise_cal=cal,
    )
    os.replace(tmp, manifest_path)

def make_output_dir():
    today = datetime.now().strftime("%Y-%m-%d")
    root = Path(f"data/{today}/{PROJECT_TAG}_{datetime.now().strftime('%Y%m%d')}")
    root.mkdir(parents=True, exist_ok=True)
    return root

def point_with_retries(telescope, alt_deg, az_deg, retries=3, retry_delay=5):
    for attempt in range(1, retries + 1):
        try:
            telescope.point(alt_deg, az_deg, wait=True)
            return True
        except AssertionError as e:
            print(f"[point] Attempt {attempt}/{retries} failed: {e}")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"[point] Attempt {attempt}/{retries} unexpected error: {e}")
            time.sleep(retry_delay)
    return False

def estimate_nspec(spec, integration_min):
    """
    Convert desired minutes on-source to number of spectra for leusch.Spectrometer.
    """
    tint = spec.int_time()  # seconds per spectrum, according to shared leusch.py
    nspec = max(1, int(np.ceil((integration_min * 60.0) / tint)))
    return nspec, tint

def take_noise_cal(spec, noise, outdir, point_index, l_deg, b_deg, nspec):
    """
    Take a calibration spectrum with the noise diode on at the current pointing.
    """
    cal_stamp = timestamp()
    base = f"point_{point_index:04d}_l{l_deg:06.2f}_b{b_deg:06.2f}_{cal_stamp}_noise_on"
    fits_path = outdir / f"{base}.fits"
    meta_path = outdir / f"{base}.json"

    print("[cal] Turning noise diode ON")
    noise.on()
    time.sleep(1.0)

    spec.read_spec(str(fits_path), nspec, (l_deg, b_deg), system=COORD_SYSTEM)

    print("[cal] Turning noise diode OFF")
    noise.off()

    meta = {
        "kind": "noise_cal",
        "timestamp_utc": cal_stamp,
        "jd": jd_now(),
        "l_deg": l_deg,
        "b_deg": b_deg,
        "nspec": nspec,
        "coord_system": COORD_SYSTEM,
        "fits_file": str(fits_path),
        "noise_diode": "on",
    }
    write_json(meta_path, meta)
    return str(fits_path), str(meta_path)


# ============================================================================
# Main observing loop
# ============================================================================

def main():
    print("=" * 72)
    print("Leuschner HI Mapping — Project 12: Big High-Velocity Cloud")
    print("=" * 72)

    outdir = make_output_dir()
    manifest_path = outdir / "manifest_latest.npz"
    session_summary_path = outdir / "session_summary.json"

    print(f"[init] Output directory: {outdir}")

    print("[init] Connecting to telescope...")
    telescope = leusch.LeuschTelescope()

    print("[init] Connecting to spectrometer...")
    spec = leusch.Spectrometer()
    spec.check_connected()

    print("[init] Connecting to noise diode...")
    noise = leusch.LeuschNoise()

    nspec_on, tint = estimate_nspec(spec, INTEGRATION_MIN)
    print(f"[init] Spectrometer integration time per spectrum: {tint:.3f} s")
    print(f"[init] Using nspec = {nspec_on} for {INTEGRATION_MIN:.1f} min on-source")

    targets = build_hvc_grid()
    print(f"[init] Built {len(targets)} Galactic grid targets")

    records = []
    n_done = 0
    n_cal = 0

    # Try current pointing first; if unavailable, stow-ish fallback
    try:
        cur_alt, cur_az = telescope.get_pointing()
        print(f"[init] Current pointing: alt={cur_alt:.2f} deg, az={cur_az:.2f} deg")
    except Exception:
        cur_alt, cur_az = 45.0, 180.0
        print("[init] Could not query current pointing; using fallback alt=45, az=180")

    try:
        while True:
            choice = choose_next_target(targets, cur_alt, cur_az)

            if choice is None:
                remaining = sum(not t["observed"] for t in targets)
                if remaining == 0:
                    print("[main] All targets completed.")
                    break

                print("[main] No valid visible targets right now. Waiting...")
                time.sleep(RECHECK_VISIBLE_SEC)
                try:
                    cur_alt, cur_az = telescope.get_pointing()
                except Exception:
                    pass
                continue

            tgt, alt_deg, az_deg = choice
            l_deg = tgt["l_deg"]
            b_deg = tgt["b_deg"]

            print("-" * 72)
            print(
                f"[main] Next target id={tgt['id']}  "
                f"(l,b)=({l_deg:.2f}, {b_deg:.2f})  "
                f"-> (alt,az)=({alt_deg:.2f}, {az_deg:.2f})"
            )

            ok = point_with_retries(telescope, alt_deg, az_deg)
            if not ok:
                print("[main] Pointing failed; will retry this target later.")
                time.sleep(5)
                continue

            time.sleep(SETTLE_SEC)
            cur_alt, cur_az = alt_deg, az_deg

            used_noise_cal = False
            if (n_done % CAL_EVERY_N_POINTS) == 0:
                try:
                    take_noise_cal(
                        spec=spec,
                        noise=noise,
                        outdir=outdir,
                        point_index=tgt["id"],
                        l_deg=l_deg,
                        b_deg=b_deg,
                        nspec=CAL_NSPEC,
                    )
                    used_noise_cal = True
                    n_cal += 1
                except Exception as e:
                    print(f"[cal] Warning: noise calibration failed: {e}")

            obs_stamp = timestamp()
            base = f"point_{tgt['id']:04d}_l{l_deg:06.2f}_b{b_deg:06.2f}_{obs_stamp}"
            fits_path = outdir / f"{base}.fits"
            meta_path = outdir / f"{base}.json"

            print(f"[obs] Recording to {fits_path.name}")
            spec.read_spec(str(fits_path), nspec_on, (l_deg, b_deg), system=COORD_SYSTEM)

            record = {
                "id": int(tgt["id"]),
                "timestamp_utc": obs_stamp,
                "jd": float(jd_now()),
                "l_deg": float(l_deg),
                "b_deg": float(b_deg),
                "alt_deg": float(alt_deg),
                "az_deg": float(az_deg),
                "coord_system": COORD_SYSTEM,
                "nspec": int(nspec_on),
                "spec_int_time_sec": float(tint),
                "integration_min_requested": float(INTEGRATION_MIN),
                "used_noise_cal": bool(used_noise_cal),
                "fits_file": str(fits_path),
            }

            write_json(meta_path, record)

            tgt["observed"] = True
            records.append(record)
            n_done += 1

            update_manifest_npz(manifest_path, records)

            summary = {
                "project": "Project 12: Big High-Velocity Cloud",
                "output_dir": str(outdir),
                "n_targets_total": len(targets),
                "n_targets_completed": n_done,
                "n_noise_cals": n_cal,
                "integration_min_per_point": INTEGRATION_MIN,
                "spec_int_time_sec": tint,
                "nspec_per_point": nspec_on,
                "last_completed_target_id": int(tgt["id"]),
                "last_completed_l_deg": float(l_deg),
                "last_completed_b_deg": float(b_deg),
                "last_completed_utc": obs_stamp,
            }
            write_json(session_summary_path, summary)

            print(
                f"[save] Done. Completed {n_done} / {len(targets)} targets. "
                f"Manifest updated: {manifest_path.name}"
            )

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        print("[main] Attempting to turn noise diode OFF.")
        try:
            noise.off()
        except Exception:
            pass

        print("[main] Stowing telescope.")
        try:
            telescope.stow(wait=True)
        except Exception as e:
            print(f"[main] Warning: failed to stow telescope: {e}")

        print("[main] Session ended.")
        print(f"[main] Data directory: {outdir}")


if __name__ == "__main__":
    main()
