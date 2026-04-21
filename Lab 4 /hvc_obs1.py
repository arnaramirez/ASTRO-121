import os
import csv
import time
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

from leusch_module import LeuschTelescope, Spectrometer

# --------------------------------------------------
# Site coordinates from leo.py
# --------------------------------------------------
SITE = EarthLocation(lat=37.9183 * u.deg, lon=-122.1067 * u.deg, height=304 * u.m)

# Telescope safety bounds
ALT_MIN = 15.0
ALT_MAX = 85.0
AZ_MIN = 5.0
AZ_MAX = 350.0

# Output directory (UTC date)
today_utc = Time.now().utc.strftime("%Y-%m-%d")
OUTDIR = Path(f"data/{today_utc}")
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTDIR / "observation_log.csv"

# Number of spectra to accumulate per pointing
NSPEC = 60  # placeholder; set from desired integration time

# How long to wait if nothing is currently observable
SLEEP_IF_NONE_VISIBLE = 300  # seconds


def make_serpentine_grid(l_min=60, l_max=180, b_min=20, b_max=60, dl=4, db=4):
    """
    Build a serpentine grid in Galactic coordinates.

    Returns
    -------
    targets : list of tuples
        [(l_deg, b_deg), ...]
    """
    l_vals = np.arange(l_min, l_max + 0.1, dl)
    b_vals = np.arange(b_min, b_max + 0.1, db)

    targets = []
    for i, b in enumerate(b_vals):
        l_row = l_vals if i % 2 == 0 else l_vals[::-1]
        for l in l_row:
            targets.append((float(l), float(b)))
    return targets


def gal_to_altaz(l_deg, b_deg, obstime):
    """
    Convert Galactic coordinates to ICRS and Alt/Az for a given observing time.

    Parameters
    ----------
    l_deg, b_deg : float
        Galactic longitude and latitude in degrees.
    obstime : astropy.time.Time
        Observation time.

    Returns
    -------
    ra_deg, dec_deg, alt_deg, az_deg : floats
    """
    gal = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    icrs = gal.icrs
    altaz = gal.transform_to(AltAz(obstime=obstime, location=SITE))
    return icrs.ra.deg, icrs.dec.deg, altaz.alt.deg, altaz.az.deg


def observable(alt, az):
    """
    Check whether a target is safely observable.
    """
    return (ALT_MIN < alt < ALT_MAX) and (AZ_MIN < az < AZ_MAX)


def append_log(row):
    """
    Append one row to the CSV observation log.
    """
    exists = LOGFILE.exists()
    with open(LOGFILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    tel = LeuschTelescope()
    spec = Spectrometer()

    targets = make_serpentine_grid()
    remaining = targets.copy()

    try:
        tint = spec.int_time()
        print(f"Integration time per spectrum: {tint:.3f} s")
        print(f"Total time per pointing: {NSPEC * tint / 60:.2f} min")
    except Exception as e:
        print("Could not get spectrometer integration time:", e)
        tint = None

    while remaining:
        # Use one shared astropy time object for this scheduling pass
        now = Time.now()

        visible = []
        for idx, (l_deg, b_deg) in enumerate(remaining):
            ra_deg, dec_deg, alt_deg, az_deg = gal_to_altaz(l_deg, b_deg, now)
            if observable(alt_deg, az_deg):
                visible.append((idx, l_deg, b_deg, ra_deg, dec_deg, alt_deg, az_deg))

        if not visible:
            print(f"No targets currently observable. Sleeping {SLEEP_IF_NONE_VISIBLE} s.")
            time.sleep(SLEEP_IF_NONE_VISIBLE)
            continue

        # Choose the first visible target in the precomputed serpentine order
        idx, l_deg, b_deg, ra_deg, dec_deg, alt_deg, az_deg = visible[0]

        # Use one shared observation timestamp for filename + log
        obs_time = Time.now()
        stamp = obs_time.utc.strftime("%Y%m%dT%H%M%S")

        fname = OUTDIR / f"hvc_l{int(round(l_deg)):03d}_b{int(round(b_deg)):03d}_{stamp}.fits"

        print(f"\nObserving l={l_deg:.1f}, b={b_deg:.1f}")
        print(f"  RA={ra_deg:.2f} deg, Dec={dec_deg:.2f} deg")
        print(f"  Alt={alt_deg:.2f} deg, Az={az_deg:.2f} deg")
        print(f"  UTC={obs_time.isot}, JD={obs_time.jd:.8f}")

        try:
            tel.point(alt_deg, az_deg, wait=True)
            alt_now, az_now = tel.get_pointing()

            spec.read_spec(str(fname), NSPEC, (l_deg, b_deg), system="ga")

            append_log({
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "filename": str(fname),
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "alt_deg_cmd": alt_deg,
                "az_deg_cmd": az_deg,
                "alt_deg_actual": alt_now,
                "az_deg_actual": az_now,
                "nspec": NSPEC,
                "int_time_each_s": tint if tint is not None else "",
                "total_int_time_s": (NSPEC * tint) if tint is not None else "",
                "status": "ok",
            })

            remaining.pop(idx)
            print(f"Saved: {fname}")
            print(f"Remaining targets: {len(remaining)}")

        except Exception as e:
            print("Observation failed:", e)

            append_log({
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "filename": str(fname),
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "alt_deg_cmd": alt_deg,
                "az_deg_cmd": az_deg,
                "alt_deg_actual": "",
                "az_deg_actual": "",
                "nspec": NSPEC,
                "int_time_each_s": tint if tint is not None else "",
                "total_int_time_s": (NSPEC * tint) if tint is not None else "",
                "status": f"failed: {e}",
            })

            time.sleep(10)

    print("All targets completed.")


if __name__ == "__main__":
    main()
