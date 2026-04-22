import csv
import time
from pathlib import Path

import numpy as np
import ugradio
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

from leusch_module import LeuschTelescope, Spectrometer

# --------------------------------------------------
# Site coordinates from ugradio.leo
# --------------------------------------------------
SITE = EarthLocation(
    lat=ugradio.leo.lat * u.deg,
    lon=ugradio.leo.lon * u.deg,
    height=ugradio.leo.alt * u.m,
)

# Telescope safety bounds
ALT_MIN = 15.0
ALT_MAX = 85.0
AZ_MIN = 5.0
AZ_MAX = 350.0

# Output directory (UTC date)
today_utc = Time.now().utc.strftime("%Y-%m-%d")
OUTDIR = Path(f"data/{today_utc}").resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTDIR / "observation_log.csv"

# Number of spectra to accumulate per pointing
NSPEC = 60  # adjust as needed

# Sleep time if no targets are currently observable
SLEEP_IF_NONE_VISIBLE = 300  # seconds


def make_serpentine_grid(l_min=60, l_max=180, b_min=20, b_max=60, dl=4, db=4):
    l_vals = np.arange(l_min, l_max + 0.1, dl)
    b_vals = np.arange(b_min, b_max + 0.1, db)

    targets = []
    for i, b in enumerate(b_vals):
        l_row = l_vals if i % 2 == 0 else l_vals[::-1]
        for l in l_row:
            targets.append((float(l), float(b)))
    return targets


def gal_to_altaz(l_deg, b_deg, obstime):
    gal = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    icrs = gal.icrs
    altaz = gal.transform_to(AltAz(obstime=obstime, location=SITE))
    return icrs.ra.deg, icrs.dec.deg, altaz.alt.deg, altaz.az.deg


def observable(alt_deg, az_deg):
    return (ALT_MIN < alt_deg < ALT_MAX) and (AZ_MIN < az_deg < AZ_MAX)


def append_log(row):
    exists = LOGFILE.exists()
    with open(LOGFILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def choose_next_target(remaining, now):
    for idx, (l_deg, b_deg) in enumerate(remaining):
        ra_deg, dec_deg, alt_deg, az_deg = gal_to_altaz(l_deg, b_deg, now)
        if observable(alt_deg, az_deg):
            return idx, l_deg, b_deg, ra_deg, dec_deg, alt_deg, az_deg
    return None


def acquire_fits(spec, fits_path, nspec, l_deg, b_deg):
    """
    Ask the spectrometer helper to write a FITS file and verify it exists.
    """
    fits_path = Path(fits_path).resolve()
    fits_path.parent.mkdir(parents=True, exist_ok=True)

    if fits_path.exists():
        fits_path.unlink()

    print(f"[spec] Writing FITS to: {fits_path}")
    spec.read_spec(str(fits_path), nspec, (l_deg, b_deg), system="ga")
    time.sleep(1.0)

    if not fits_path.exists():
        raise RuntimeError(f"FITS file was not created: {fits_path}")
    if fits_path.stat().st_size == 0:
        raise RuntimeError(f"FITS file is empty: {fits_path}")

    print(f"[spec] FITS verified: {fits_path} ({fits_path.stat().st_size} bytes)")
    return fits_path


def fits_to_npz(fits_path, npz_path, metadata):
    """
    Read the FITS file and save a companion .npz file using np.savez.

    This assumes the spectral tables contain columns like:
    auto0_real, auto1_real, cross_real, cross_imag
    based on your Spectrometer docstring.
    """
    fits_path = Path(fits_path).resolve()
    npz_path = Path(npz_path).resolve()

    auto0_list = []
    auto1_list = []
    cross_real_list = []
    cross_imag_list = []

    with fits.open(fits_path) as hdul:
        # HDU 0 is usually primary header, tables start after that
        for hdu in hdul[1:]:
            data = hdu.data
            if data is None:
                continue

            names = set(data.names) if data.names is not None else set()

            if "auto0_real" in names:
                auto0_list.append(np.array(data["auto0_real"]))
            if "auto1_real" in names:
                auto1_list.append(np.array(data["auto1_real"]))
            if "cross_real" in names:
                cross_real_list.append(np.array(data["cross_real"]))
            if "cross_imag" in names:
                cross_imag_list.append(np.array(data["cross_imag"]))

    # Convert lists to arrays when present
    save_dict = dict(metadata)

    if auto0_list:
        save_dict["auto0_real"] = np.array(auto0_list)
    if auto1_list:
        save_dict["auto1_real"] = np.array(auto1_list)
    if cross_real_list:
        save_dict["cross_real"] = np.array(cross_real_list)
    if cross_imag_list:
        save_dict["cross_imag"] = np.array(cross_imag_list)

    if not any(k in save_dict for k in ["auto0_real", "auto1_real", "cross_real", "cross_imag"]):
        raise RuntimeError(f"No expected spectral columns found in FITS: {fits_path}")

    np.savez(npz_path, **save_dict)

    if not npz_path.exists() or npz_path.stat().st_size == 0:
        raise RuntimeError(f"NPZ file was not created correctly: {npz_path}")

    print(f"[save] NPZ saved: {npz_path} ({npz_path.stat().st_size} bytes)")
    return npz_path


def main():
    print("=" * 60)
    print("Leuschner HI HVC Mapping Script")
    print("=" * 60)
    print(f"Site latitude : {ugradio.leo.lat:.4f} deg")
    print(f"Site longitude: {ugradio.leo.lon:.4f} deg")
    print(f"Site altitude : {ugradio.leo.alt:.1f} m")
    print(f"Output dir    : {OUTDIR}")
    print()

    tel = LeuschTelescope()
    spec = Spectrometer()

    targets = make_serpentine_grid()
    remaining = targets.copy()

    try:
        tint = spec.int_time()
        print(f"Integration time per spectrum: {tint:.3f} s")
        print(f"Total integration per pointing: {NSPEC * tint / 60:.2f} min")
    except Exception as e:
        print("Could not get spectrometer integration time:", e)
        tint = None

    while remaining:
        now = Time.now()
        chosen = choose_next_target(remaining, now)

        if chosen is None:
            print(f"No targets currently observable. Sleeping {SLEEP_IF_NONE_VISIBLE} s.")
            time.sleep(SLEEP_IF_NONE_VISIBLE)
            continue

        idx, l_deg, b_deg, ra_deg, dec_deg, alt_deg, az_deg = chosen

        obs_time = Time.now()
        stamp = obs_time.utc.strftime("%Y%m%dT%H%M%S")

        fits_name = OUTDIR / f"hvc_l{int(round(l_deg)):03d}_b{int(round(b_deg)):03d}_{stamp}.fits"
        npz_name = OUTDIR / f"hvc_l{int(round(l_deg)):03d}_b{int(round(b_deg)):03d}_{stamp}.npz"

        print(f"\nObserving l={l_deg:.1f}, b={b_deg:.1f}")
        print(f"  RA={ra_deg:.3f} deg, Dec={dec_deg:.3f} deg")
        print(f"  Alt={alt_deg:.3f} deg, Az={az_deg:.3f} deg")
        print(f"  UTC={obs_time.isot}")
        print(f"  JD ={obs_time.jd:.8f}")

        try:
            tel.point(alt_deg, az_deg, wait=True)
            alt_now, az_now = tel.get_pointing()

            fits_path = acquire_fits(spec, fits_name, NSPEC, l_deg, b_deg)

            metadata = {
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "alt_deg_cmd": alt_deg,
                "az_deg_cmd": az_deg,
                "alt_deg_actual": alt_now,
                "az_deg_actual": az_now,
                "nspec": NSPEC,
                "int_time_each_s": tint if tint is not None else np.nan,
                "total_int_time_s": (NSPEC * tint) if tint is not None else np.nan,
                "source_fits": str(fits_path),
            }

            npz_path = fits_to_npz(fits_path, npz_name, metadata)

            append_log({
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "filename_npz": str(npz_path),
                "filename_fits": str(fits_path),
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
                "npz_size_bytes": npz_path.stat().st_size,
                "fits_size_bytes": fits_path.stat().st_size,
                "status": "ok",
            })

            remaining.pop(idx)
            print(f"Remaining targets: {len(remaining)}")

        except Exception as e:
            print("[error] Observation failed:", e)

            append_log({
                "utc_time": obs_time.isot,
                "jd": obs_time.jd,
                "filename_npz": str(npz_name),
                "filename_fits": str(fits_name),
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
                "npz_size_bytes": "",
                "fits_size_bytes": "",
                "status": f"failed: {e}",
            })

            time.sleep(10)

    print("All targets completed.")


if __name__ == "__main__":
    main()
