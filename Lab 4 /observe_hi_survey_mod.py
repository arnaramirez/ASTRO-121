"""
observe_hi_survey.py
--------------------
HI 21-cm Galactic Plane Survey with the Leuschner Dish
Region: l=60deg-180deg, b=20deg-60deg (~3700 sq deg, ~900 pointings at 2deg spacing)
Goal:   3-D map of HI emission in (l, b, v_LSR) space

Observing strategy:
  - Dual SDR (pol 0 & 1) at each pointing
  - In-band frequency switching (+/- freq_offset) to separate signal from
    instrument spectral baseline
  - Periodic noise-diode injections for Tsys calibration
  - Tracks each pointing while integrating; skips if below horizon
  - Completed pointings logged to a JSON progress file to allow resuming

Usage:
    python3 observe_hi_survey.py [--plan] [--resume] [--dry-run] [--mode]

    --plan    : Print visibility windows for today and exit
    --resume  : Load progress file and skip completed pointings
    --dry-run : Step through grid without moving telescope or capturing data
    --mode : What do you want to observe (default is HVC)

Output per pointing (in data/<date>/):
    HI_l<LLL.L>_b<+/-BB.B>_jd<XXXXXXXXX.XXXXX>_lo<FREQ>_<tag>.npz
"""

import argparse
import json
import time
import threading
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.signal import medfilt

# -- ugradio imports ----------------------------------------------------------
import ugradio
from ugradio import leo, leusch, timing, coord

# -- Thread safety -----------------------------------------------------------
# RTL-SDR USB calls can deadlock if two SDRs hit librtlsdr simultaneously.
# This lock serialises all SDR hardware calls across threads.
# Each SDR gets its own lock since they are independent USB devices.
_sdr0_lock = threading.Lock()
_sdr1_lock = threading.Lock()

# Global libusb lock -- serialises ALL sdr.capture_data() calls across
# both SDRs. libusb uses a global mutex internally and crashes with an
# assertion failure if two threads hit it simultaneously even on different
# USB devices. This lock ensures only one capture_data call runs at a time.
_usb_lock = threading.Lock()

# Dish lock: prevents tracker from sending a point() command while
# observe_pointing is in the middle of the initial point/settle sequence.
_dish_lock = threading.Lock()

# -- Site constants (Leuschner Observatory) ----------------------------------
LAT_DEG  =  leo.lat   # degrees N
LON_DEG  = leo.lon  # degrees E  (ugradio uses East-positive)
ALT_M    =  leo.alt     # meters

# -- Receiver / SDR constants -------------------------------------------------
HI_FREQ      = 1420.405751e6   # HI rest frequency [Hz]
C_MS         = 299792458.0   # speed of light [m/s]

SAMPLE_RATE  = 2.2e6           # [Hz]  2.2 MHz -- actual rate the Pi RTL-SDR supports
NSAMPLES_FFT = 2**10           # 1024 pts -> deltanu ~= 2148 Hz -> deltav ~= 0.45 km/s
                               # 22x faster FFT than 2**14, still sub-km/s resolution for HVC
# Integration time per pointing:
#   one block = NSAMPLES_FFT / SAMPLE_RATE = 16384 / 2.5e6 ~= 6.55 ms
#   NBLOCKS_OBS blocks per freq-switch position x 2 positions x 2 pols
#   -> total wall time ~= NBLOCKS_OBS x 6.55 ms x 4 ~= NBLOCKS_OBS x 26 ms
#   NBLOCKS_OBS = 4096  ->  ~107 s/position ->  ~7 min total per pointing  (current)  (ok)
#   NBLOCKS_OBS = 4096  ->  ~107 s/position ->  ~7 min total per pointing  (current)
NBLOCKS_OBS  = 32768           # blocks per SDR per freq-switch half
                               # 32768 x 1024 / 2.2e6 = 15s per window
                               # ON+OFF+slew ~70s/pointing -> 1281 x 70s = 25hrs
NBLOCKS_CAL  = 128             # blocks for noise-diode calibration (~1.7 s, plenty for Y-factor)
GAIN         = 20              # SDR gain [dB] -- 19.7dB actual (nearest valid step)
FREQ_OFFSET  = 0.5e6           # LO frequency-switch offset [Hz]
# DC dropout fills entire 2.2 MHz band so no placement avoids it entirely.
# 0.5 MHz offset maximises valid overlap (~+/- 42 km/s) while keeping
# HI near the band centre where the bandpass shapes best match.
# Residual baseline slope is removed in post with polynomial fitting.

# -----------------------------------------------------------------------------
# HVC VELOCITY COVERAGE
# -----------------------------------------------------------------------------
# High Velocity Clouds (HVCs) are defined as HI gas with |v_LSR| > 90 km/s.
# The standard FREQ_OFFSET=0.5 MHz only covers +/-42 km/s (standard HI).
# To detect HVCs, each pointing is observed at multiple LO centre frequencies,
# each shifted to place a different velocity window in the valid band.
#
# Each LO shift moves the centre of the observed velocity window:
#   v_centre = C_MS * (HI_FREQ - LO_centre) / HI_FREQ / 1e3  [km/s]
#
# Window definitions (centre_velocity_kms, label):
#   Each window covers centre +/- ~42 km/s of valid bandwidth
#
# Known HVC complexes in survey region (l=60-180, b=20-60):
#   Complex C:  v_LSR ~ -80 to -130 km/s  (most prominent HVC in N sky)
#   Complex A:  v_LSR ~ -150 to -200 km/s
#   IVC (Intermediate Velocity Clouds): v_LSR ~ -50 to -90 km/s
#
# Primary science goal: detect High Velocity Clouds (HVCs) at |v_LSR| > 90 km/s
# Strategy: observe each pointing at TWO velocity windows --
#   1. "std"  (v=0 km/s):    covers local HI, used as reference/context
#   2. "hvc"  (v=-150 km/s): covers Complex C/A, primary HVC target
#
# Known HVC complexes in survey region (l=60-180, b=20-60):
#   Complex C: v_LSR ~ -80 to -170 km/s  (most prominent HVC in northern sky,
#              thought to be accreting low-metallicity gas onto the Milky Way)
#   Complex A: v_LSR ~ -150 to -200 km/s
#   Complex M: v_LSR ~  -70 to -130 km/s
#
# Observing mode -- controls which velocity windows are observed:
#   "hvc_only" : HVC window only -- maximises HVC sensitivity per night
#                Best choice when HVC detection is the primary goal.
#                1281 pointings x ~2 min = ~43 hours = ~7 nights
#   "both"     : standard HI + HVC windows -- doubles time per pointing
#                Use when you also want a local HI map for context.
#                1281 pointings x ~4 min = ~85 hours = ~14 nights
#   "std_only" : standard HI only -- fastest, but misses HVCs entirely
#
# Radiometer equation analysis (Tsys=100K, delta_nu=134Hz, smooth=1km/s):
#   sigma_1kms per visit = 100 / sqrt(134 x t_int x 7) = 0.47K  (t_int=61s)
#   SNR(3K HVC)  = 3.0 / 0.47 = 6.4 sigma  -- detectable in single visit
#   SNR(1K HVC)  = 1.0 / 0.47 = 2.1 sigma  -- need ~6 stacked visits
#   SNR(0.5K HVC)= 0.5 / 0.47 = 1.1 sigma  -- need ~22 stacked visits
#
# Conclusion: focus on HVC window -- spend all time where the science is.
OBSERVE_MODE = "hvc_only"   # "hvc_only" | "both" | "std_only"

# Derived flag for backward compatibility
OBSERVE_HVC = OBSERVE_MODE in ("hvc_only", "both")

# LO centre frequency for a target velocity window centre
def vel_to_lo(v_centre_kms):
    """
    Return the LO centre frequency [Hz] that places v_centre_kms at the
    centre of the valid overlap bandwidth.

    The HI line from gas at v_centre_kms is Doppler shifted to:
        f_HI = HI_FREQ * (1 - v_centre_kms * 1e3 / C_MS)
    Setting LO = f_HI puts that gas at DC in the band (zero offset).
    The freq switching ON/OFF positions then bracket this centre by
    +/- FREQ_OFFSET, giving a valid window of +/- ~42 km/s around v_centre.
    """
    return HI_FREQ * (1.0 - v_centre_kms * 1e3 / C_MS)

# Velocity windows to observe at each pointing (label, v_centre_kms)
# Each window covers v_centre +/- (FREQ_OFFSET/HI_FREQ * c) km/s
# With FREQ_OFFSET=0.5 MHz: +/- 42 km/s per window
VELOCITY_WINDOWS = [
    ("std",  0.0),      # Local HI:    covers  -42 to  +42 km/s
    ("hvc", -150.0),    # HVC Complex C/A: covers -192 to -108 km/s
]
# Empirically check DC dropout at hvc LO before relying on this:
#   LO_ON  at hvc window = vel_to_lo(-150) - 0.5 MHz = ~1420.617 MHz
#   LO_OFF at hvc window = vel_to_lo(-150) + 0.5 MHz = ~1421.617 MHz
# These are different from the original 1421.406 MHz that had severe dropout.
# Run the LO sweep test to confirm these are clean before a full session.
#
# With OBSERVE_HVC=False only "std" is observed -- use for initial survey pass.

# Noise diode increments [K]
T_ND_POL0 = 79.0
T_ND_POL1 = 58.0

# Calibration cadence: inject noise diode every N pointings
CAL_EVERY_N = 5

# Min elevation for observing [deg]
MIN_EL = 30.0   # hard lower limit -- Leuschner can observe reliably to 30deg

# Tracker soft margin -- follows target this far below MIN_EL rather
# than freezing. Keeps dish on-target if integration runs slightly long.
# Beam efficiency at 2deg offset: exp(-4*ln2*(2/4)^2) = 0.76 -- acceptable.
TRACK_SOFT_MARGIN = 2.0   # deg below MIN_EL tracker will still follow target

# Beam FWHM ~= 4deg, Nyquist spacing = 2deg
GRID_SPACING_DEG = 2.0

# -----------------------------------------------------------------------------
# HORIZON OBSTRUCTION MASK
# -----------------------------------------------------------------------------
# Maps azimuth (0=N, 90=E, 180=S, 270=W) to the minimum CLEAR elevation above
# that direction. Pointings physically blocked by hills/buildings are skipped
# even if they are above the mathematical MIN_EL horizon.
#
# HOW TO FILL THIS IN:
#   Go outside at Leuschner and note where hills/treelines are visible.
#   For each obstructed azimuth direction, estimate the angle above flat
#   horizon that the obstruction reaches.  Set to 0.0 where sky is clear.
#
# Format: az_deg (0-360) -> min clearance elevation in degrees
# Intermediate azimuths are linearly interpolated.
#
# Current values are conservative placeholders -- update from site survey.
HORIZON_MASK = {
      0: 5.0,   # N   -- ridge to north
     45: 3.0,   # NE
     90: 2.0,   # E   -- relatively open
    135: 2.0,   # SE
    180: 3.0,   # S
    225: 5.0,   # SW
    270: 8.0,   # W   -- hills to west
    315: 10.0,  # NW  -- adjust if Lafayette hills are a problem here
    360: 5.0,   # N (same as 0 to close the circle)
}

# Set USE_HORIZON_MASK = False to disable obstruction masking entirely
# (falls back to plain MIN_EL check, original behaviour)
USE_HORIZON_MASK = True


def horizon_clearance(az_deg):
    """
    Return the minimum clear elevation [deg] at a given azimuth by
    linearly interpolating the HORIZON_MASK lookup table.
    """
    az_deg = az_deg % 360
    azimuths = sorted(HORIZON_MASK.keys())
    elevations = [HORIZON_MASK[a] for a in azimuths]

    # Wrap-around interpolation (e.g. between 315deg and 360/0deg)
    if az_deg <= azimuths[0] or az_deg >= azimuths[-1]:
        # Interpolate between last and first entry across the 360/0 boundary
        az_lo, az_hi = azimuths[-1], azimuths[0] + 360
        el_lo, el_hi = elevations[-1], elevations[0]
        t = (az_deg - az_lo) % 360 / (az_hi - az_lo)
        return el_lo + t * (el_hi - el_lo)

    # Normal interpolation
    for k in range(len(azimuths) - 1):
        if azimuths[k] <= az_deg <= azimuths[k + 1]:
            t = (az_deg - azimuths[k]) / (azimuths[k + 1] - azimuths[k])
            return elevations[k] + t * (elevations[k + 1] - elevations[k])

    return MIN_EL  # fallback


# Galactic coordinate bounds
L_MIN, L_MAX = 60.0,  180.0
B_MIN, B_MAX = 20.0,   60.0

# -- Paths --------------------------------------------------------------------
# DATA_DIR and PROGRESS_FILE are resolved dynamically at runtime so that
# multi-day sessions automatically partition into per-day subdirectories.
# Use get_day_dir() and get_progress_file() everywhere instead of constants.

BASE_DATA_DIR = Path("data")
PROGRESS_FILE = Path("survey_progress.json")   # global across all days


def get_day_dir(date_str=None):
    """
    Return (and create) the data directory for a given date string YYYY-MM-DD.
    If date_str is None, uses today's UTC date.
    Structure: data/<date>/
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    d = BASE_DATA_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_day_progress_file(date_str=None):
    """
    Per-day progress file: data/<date>/progress_<date>.json
    Tracks only the pointings observed on that specific day.
    Deleting the day folder removes both data and its progress record,
    automatically re-queuing those pointings for re-observation.
    """
    day_dir = get_day_dir(date_str)
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return day_dir / f"progress_{date_str}.json"

# -- Galactic -> Equatorial conversion (via astropy if available, else ugradio) --
try:
    from astropy.coordinates import SkyCoord, Galactic, ICRS
    import astropy.units as u

    def gal_to_radec(l_deg, b_deg):
        """Galactic (l,b) -> (RA, Dec) in degrees (J2000)."""
        c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic)
        eq = c.icrs
        return eq.ra.deg, eq.dec.deg

except ImportError:
    # Fallback: approximate matrix rotation (good to ~0.01deg)
    def gal_to_radec(l_deg, b_deg):
        """Approximate Galactic -> J2000 equatorial (degrees)."""
        l = np.radians(l_deg)
        b = np.radians(b_deg)
        # NGP: RA=192.859deg, Dec=27.128deg; ascending node l=122.932deg
        ra_ngp  = np.radians(192.8595)
        dec_ngp = np.radians(27.1284)
        l_asc   = np.radians(122.9320)

        sinb    = (np.sin(dec_ngp) * np.sin(b) +
                   np.cos(dec_ngp) * np.cos(b) * np.cos(l - l_asc))
        dec = np.arcsin(np.clip(sinb, -1, 1))

        x = np.cos(b) * np.sin(l - l_asc)
        y = (np.cos(dec_ngp) * np.sin(b) -
             np.sin(dec_ngp) * np.cos(b) * np.cos(l - l_asc))
        ra = (np.degrees(np.arctan2(x, y)) + ra_ngp * 180 / np.pi) % 360
        dec = np.degrees(dec)
        return ra, dec


# -----------------------------------------------------------------------------
# 1.  SURVEY GRID GENERATION
# -----------------------------------------------------------------------------

def make_survey_grid(l_min=L_MIN, l_max=L_MAX,
                     b_min=B_MIN, b_max=B_MAX,
                     spacing=GRID_SPACING_DEG):
    """
    Generate a list of (l, b) pointings at Nyquist spacing.
    Uses a boustrophedon (snake) pattern to minimise slew distance.
    Returns list of dicts: {l, b, ra, dec, pointing_id}
    """
    l_vals = np.arange(l_min, l_max + spacing * 0.5, spacing)
    b_vals = np.arange(b_min, b_max + spacing * 0.5, spacing)

    grid = []
    for j, b in enumerate(b_vals):
        row_l = l_vals if j % 2 == 0 else l_vals[::-1]  # snake pattern
        for l in row_l:
            ra, dec = gal_to_radec(l, b)
            grid.append(dict(
                l=round(float(l), 2),
                b=round(float(b), 2),
                ra=round(ra, 6),
                dec=round(dec, 6),
                pointing_id=f"l{l:07.3f}_b{b:+07.3f}"
            ))

    print(f"[grid] Survey grid: {len(grid)} pointings "
          f"({len(l_vals)} in l x {len(b_vals)} in b)")
    return grid


# -----------------------------------------------------------------------------
# 2.  PROGRESS TRACKING  (two-level: global + per-day)
# -----------------------------------------------------------------------------
#
# Global progress  (survey_progress.json at repo root)
#   - Union of all completed pointings across all days.
#   - Used by the scheduler to know what still needs observing overall.
#
# Per-day progress  (data/<date>/progress_<date>.json)
#   - Records only pointings completed on that calendar day.
#   - Lives inside the day's data folder -- deleting the folder wipes both
#     the data AND the day's progress, automatically re-queuing those
#     pointings so they will be re-observed on the next session.
#
# On each successful observation both files are updated atomically.

def _atomic_json_write(path, data):
    """Write a JSON file atomically via a temp file + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_global_progress():
    """Return set of ALL completed pointing_ids across every session."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        completed = set(data.get("completed", []))
        print(f"[progress] Global: {len(completed)} pointings completed so far.")
        return completed
    return set()


def load_day_progress(date_str=None):
    """Return set of pointing_ids completed on a specific day."""
    pfile = get_day_progress_file(date_str)
    if pfile.exists():
        with open(pfile) as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def save_progress(global_set, day_set, date_str=None):
    """
    Atomically update both the global and per-day progress files.
    Call after every successful pointing observation.
    """
    _atomic_json_write(
        PROGRESS_FILE,
        {"completed": sorted(global_set),
         "last_updated": datetime.utcnow().isoformat()}
    )
    _atomic_json_write(
        get_day_progress_file(date_str),
        {"date": date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d'),
         "completed": sorted(day_set),
         "last_updated": datetime.utcnow().isoformat()}
    )


def rebuild_global_progress():
    """
    Reconstruct the global progress file by scanning all per-day progress files.
    Useful after deleting a day folder to re-sync the global record.

    Run manually:
        python3 -c "from observe_hi_survey import rebuild_global_progress; rebuild_global_progress()"
    """
    all_completed = set()
    for pfile in sorted(BASE_DATA_DIR.glob("*/progress_*.json")):
        with open(pfile) as f:
            day_data = json.load(f)
        day_ids = set(day_data.get("completed", []))
        all_completed |= day_ids
        print(f"[rebuild] {pfile.parent.name}: {len(day_ids)} pointings")

    _atomic_json_write(
        PROGRESS_FILE,
        {"completed": sorted(all_completed),
         "last_updated": datetime.utcnow().isoformat(),
         "note": "rebuilt from per-day progress files"}
    )
    print(f"[rebuild] Global progress rebuilt: {len(all_completed)} total pointings.")


# -----------------------------------------------------------------------------
# 3.  VISIBILITY PLANNING
# -----------------------------------------------------------------------------

def pointing_is_visible(ra_deg, dec_deg, jd, min_el=MIN_EL):
    """
    Return (alt, az) if pointing is above min_el and unobstructed, else None.
    Single hard threshold -- no buffer stages or future checks.
    """
    alt, az = coord.get_altaz(ra_deg, dec_deg, jd,
                              LAT_DEG, LON_DEG, ALT_M,
                              equinox='J2000')
    if alt < min_el:
        return None
    if USE_HORIZON_MASK:
        clearance = horizon_clearance(az)
        if alt < clearance:
            return None
    return alt, az


def elevation_above_local_horizon(alt, az):
    """
    Return how many degrees a pointing clears the local obstruction profile.
    Positive = clear sky above the hill/treeline.
    Negative = physically blocked (should not be observed).
    Used by the scheduler to rank pointings by true sky clearance.
    """
    if USE_HORIZON_MASK:
        return alt - horizon_clearance(az)
    return alt - MIN_EL


def print_visibility_windows(grid, jd_start=None, hours=12, step_min=10):
    """
    Print a table of when each pointing is above MIN_EL over the next `hours`.
    Useful for scheduling sessions.
    """
    if jd_start is None:
        jd_start = timing.julian_date()

    jd_step = step_min / (24 * 60)
    jd_end  = jd_start + hours / 24
    jds     = np.arange(jd_start, jd_end, jd_step)

    # Convert to PST (UTC-8 / UTC-7 for PDT)
    def jd_to_pst(jd):
        unix = (jd - 2440587.5) * 86400
        dt   = datetime.fromtimestamp(unix)
        return dt.strftime('%H:%M')

    print(f"\n{'Pointing':>22}  {'l':>6}  {'b':>6}  "
          f"{'Rise (PST)':>12}  {'Set (PST)':>12}  {'Max El':>8}")
    print("-" * 80)

    for pt in grid:
        visible_jds = []
        max_el = 0.0
        for jd in jds:
            res = pointing_is_visible(pt['ra'], pt['dec'], jd)
            if res:
                visible_jds.append(jd)
                max_el = max(max_el, res[0])

        if visible_jds:
            rise = jd_to_pst(visible_jds[0])
            sett = jd_to_pst(visible_jds[-1])
            print(f"{pt['pointing_id']:>22}  {pt['l']:>6.2f}  {pt['b']:>+6.2f}  "
                  f"{rise:>12}  {sett:>12}  {max_el:>7.1f}deg")
        else:
            print(f"{pt['pointing_id']:>22}  {pt['l']:>6.2f}  {pt['b']:>+6.2f}  "
                  f"{'-- not visible --':>28}")


# -----------------------------------------------------------------------------
# 4.  SDR HELPERS
# -----------------------------------------------------------------------------

def build_freqs(center_hz, nfft=NSAMPLES_FFT, rate=SAMPLE_RATE):
    """Return RF frequency array [Hz] for a given SDR centre frequency."""
    baseband = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / rate))
    return baseband + center_hz


# Streaming FFT accumulation parameters.
# Capture STREAM_BATCH raw samples at a time, FFT each immediately and
# accumulate into the power spectrum. This avoids large memory allocations
# while keeping USB call overhead low.
# STREAM_BATCH x nsamples x 8 bytes (complex64) = memory per transfer:
#   STREAM_BATCH=16: 16 x 16384 x 8 = 2 MB  -- very safe on Pi
#   STREAM_BATCH=32: 32 x 16384 x 8 = 4 MB  -- still fine
STREAM_BATCH = 512  # blocks per USB transfer
                    # 512 x 1024 samples x 8 bytes = 4 MB -- safe on Pi
                    # larger batch = fewer USB calls = faster throughput


def capture_spectrum(sdr, center_hz, nblocks, gain=GAIN,
                     nsamples=NSAMPLES_FFT, max_retries=3):
    """
    Capture nblocks at center_hz using streaming FFT accumulation.

    Fetches STREAM_BATCH raw sample blocks at a time, immediately computes
    and accumulates the FFT power spectrum, then discards the raw samples.
    This keeps memory usage at ~2 MB per transfer regardless of nblocks,
    and builds up the averaged spectrum incrementally -- equivalent to
    stacking nblocks individual FFTs but with much lower peak memory.

    Benefits over large batch capture:
      - Memory: 2 MB per transfer vs 64-2000 MB for bulk capture
      - Streaming: FFT runs continuously, no gap between capture and transform
      - Resilient: each small batch can retry independently on USB errors
    """
    with _usb_lock:
        sdr.set_center_freq(center_hz)  # only freq changes per call
    # set_sample_rate and set_gain fixed at init -- not repeated


    # Accumulate power spectrum in float64 to avoid overflow over many blocks
    power_acc   = np.zeros(nsamples, dtype=np.float64)
    blocks_done = 0

    while blocks_done < nblocks:
        batch = min(STREAM_BATCH, nblocks - blocks_done)

        for attempt in range(1, max_retries + 1):
            try:
                # Fetch small batch of raw samples
                with _usb_lock:
                    result = sdr.capture_data(nsamples=nsamples, nblocks=batch)
                v      = _parse_voltages(result, nsamples)   # (batch, nsamples)

                # FFT each block and accumulate power immediately
                # Raw samples discarded after FFT -- no large array retained
                for block in v:
                    spectrum   = np.fft.fft(block.astype(np.complex64))
                    power_acc += np.abs(spectrum) ** 2

                blocks_done += batch
                break   # success

            except Exception as e:
                print(f"[sdr] stream attempt {attempt}/{max_retries} "
                      f"at block {blocks_done}: {e}")
                if attempt < max_retries:
                    time.sleep(0.3 * attempt)
                else:
                    raise RuntimeError(
                        f"capture_spectrum failed after {max_retries} attempts "
                        f"at block {blocks_done}: {e}"
                    )

    power = np.fft.fftshift(power_acc / nblocks)
    rf    = build_freqs(center_hz, nsamples)
    return rf, power


def _parse_voltages(result, nsamples):
    """
    Convert raw SDR output to complex array shape (nblocks, nsamples).
    sdr.capture_data() instance method returns (nblocks, nsamples, 2)
    where [...,0]=I and [...,1]=Q.
    """
    v = np.asarray(result)
    if v.ndim == 3 and v.shape[2] == 2:
        return (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)
    if v.ndim == 2:
        return v.astype(np.complex64)
    raise ValueError(f"Unexpected SDR output shape: {v.shape}")



# -----------------------------------------------------------------------------
# 5.  FREQUENCY-SWITCHING  (removes spectral baseline)
# -----------------------------------------------------------------------------

# Single LO switch: capture all ON blocks, then all OFF blocks.
# Interleaved switching caused constant PLL relocking which hammered the
# USB bus and caused lockups. The residual baseline slope from block
# switching is handled cleanly in post-processing with a polynomial fit
# (see analyse_hi_survey.py) -- this is more reliable than fighting the
# hardware with frequent LO changes.

# Single settle after each LO change -- two switches total per pointing
PLL_SETTLE_SEC = 0.25   # generous settle time for stable PLL lock


def freq_switch_pair(sdr, center_hz, offset_hz, nblocks,
                     nsamples=NSAMPLES_FFT,
                     quit_event=None):
    """
    Capture block-switched ON/OFF frequency-switched spectra.

    Captures all nblocks at LO_ON first, then all nblocks at LO_OFF.
    Only two LO switches per pointing -- avoids USB hammering that caused
    PLL lockup loops with interleaved switching.

    The residual bandpass slope between ON and OFF is removed in
    post-processing using a polynomial baseline fit anchored to the
    line-free edges of the spectrum.

    quit_event is checked once between ON and OFF captures for a clean stop.

    Returns:
        rf_hz            : RF frequency axis for ON position
        T_diff           : ON - OFF difference spectrum
        spec_on          : averaged ON spectrum
        spec_off_aligned : averaged OFF spectrum    """
    # Standard convention: ON below HI_FREQ, OFF above HI_FREQ
    lo_on  = center_hz - offset_hz
    lo_off = center_hz + offset_hz

    # Check quit before starting
    if quit_event is not None and quit_event.is_set():
        raise RuntimeError("freq_switch_pair: quit before capture started")

    # --- All ON blocks (one PLL settle before capture starts) ---
    print(f"[freq_switch] ON  position ({lo_on/1e6:.4f} MHz)...")
    with _usb_lock:
        sdr.set_center_freq(lo_on)
    # sample_rate and gain constant -- only center_freq changes

    time.sleep(PLL_SETTLE_SEC)   # single settle for this LO position
    rf_on,  spec_on  = capture_spectrum(sdr, lo_on,  nblocks, nsamples=nsamples)

    # Check quit between ON and OFF
    if quit_event is not None and quit_event.is_set():
        print("[freq_switch] Quit received after ON capture -- skipping OFF")
        raise RuntimeError("freq_switch_pair: quit between ON and OFF captures")

    # --- All OFF blocks (one PLL settle before capture starts) ---
    print(f"[freq_switch] OFF position ({lo_off/1e6:.4f} MHz)...")
    with _usb_lock:
        sdr.set_center_freq(lo_off)
    # sample_rate and gain constant -- only center_freq changes

    time.sleep(PLL_SETTLE_SEC)   # single settle for this LO position
    rf_off, spec_off = capture_spectrum(sdr, lo_off, nblocks, nsamples=nsamples)

    # Return raw ON and OFF on their own native frequency grids.
    # Alignment is done in post-processing where the full context is available.
    # Previously interp1d was used here but it introduced NaNs wherever the
    # OFF grid extended beyond the ON grid edges -- corrupting nearly half
    # the spec_off_pol0 array. Raw grids are clean and give more flexibility.
    # T_diff is computed over the overlap region only in post-processing.
    return rf_on, spec_on, rf_off, spec_off


# -----------------------------------------------------------------------------
# 5b. PARALLEL DUAL-SDR CAPTURE HELPER
# -----------------------------------------------------------------------------

def freq_switch_pair_parallel(sdr0, sdr1, center_hz, offset_hz, nblocks,
                               nsamples=NSAMPLES_FFT,
                               quit_event=None):
    """
    Capture ON and OFF spectra on both SDRs in parallel threads.
    Each SDR runs sequentially within its own thread (no producer-consumer).
    _usb_lock serialises all USB calls so only one libusb call runs at a time.

    Both threads share _usb_lock so they effectively take turns on USB,
    but the FFT accumulation (no USB) runs truly in parallel, and the
    PLL settle sleeps also overlap -- saving ~50% time vs fully sequential.
    """
    lo_on  = center_hz - offset_hz
    lo_off = center_hz + offset_hz

    results = {}
    errors  = {}

    def capture_sdr(key, sdr):
        try:
            # ON position
            with _usb_lock:
                sdr.set_center_freq(lo_on)
            time.sleep(PLL_SETTLE_SEC)
            with _usb_lock:
                sdr.capture_data(nsamples=nsamples, nblocks=4)

            on_acc = np.zeros(nsamples, dtype=np.float64)
            blocks_done = 0
            while blocks_done < nblocks:
                if quit_event is not None and quit_event.is_set():
                    raise RuntimeError(f"quit during {key} ON")
                batch = min(STREAM_BATCH, nblocks - blocks_done)
                with _usb_lock:
                    raw = sdr.capture_data(nsamples=nsamples, nblocks=batch)
                v = _parse_voltages(raw, nsamples)
                for block in v:
                    on_acc += np.abs(np.fft.fft(block.astype(np.complex64))) ** 2
                blocks_done += batch

            spec_on = np.fft.fftshift(on_acc / nblocks)
            rf_on   = build_freqs(lo_on, nsamples)

            # OFF position
            with _usb_lock:
                sdr.set_center_freq(lo_off)
            time.sleep(PLL_SETTLE_SEC)
            with _usb_lock:
                sdr.capture_data(nsamples=nsamples, nblocks=4)

            off_acc = np.zeros(nsamples, dtype=np.float64)
            blocks_done = 0
            while blocks_done < nblocks:
                if quit_event is not None and quit_event.is_set():
                    raise RuntimeError(f"quit during {key} OFF")
                batch = min(STREAM_BATCH, nblocks - blocks_done)
                with _usb_lock:
                    raw = sdr.capture_data(nsamples=nsamples, nblocks=batch)
                v = _parse_voltages(raw, nsamples)
                for block in v:
                    off_acc += np.abs(np.fft.fft(block.astype(np.complex64))) ** 2
                blocks_done += batch

            spec_off = np.fft.fftshift(off_acc / nblocks)
            rf_off   = build_freqs(lo_off, nsamples)
            results[key] = (rf_on, spec_on, rf_off, spec_off)

        except Exception as e:
            errors[key] = e

    # Both SDRs in parallel threads -- staggered 0.25s to avoid simultaneous init
    t0 = threading.Thread(target=capture_sdr, args=('pol0', sdr0), daemon=True)
    t1 = threading.Thread(target=capture_sdr, args=('pol1', sdr1), daemon=True)
    t0.start()
    time.sleep(0.25)
    t1.start()
    t0.join()
    t1.join()

    if errors:
        for key, e in errors.items():
            print(f"[parallel] ERROR on {key}: {e}")
        raise RuntimeError(f"Parallel SDR capture failed: {errors}")

    rf_on0, on0, rf_off0, off0 = results['pol0']
    rf_on1, on1, rf_off1, off1 = results['pol1']
    return rf_on0, on0, rf_off0, off0, rf_on1, on1, rf_off1, off1


def capture_spectrum_parallel(sdr0, sdr1, center_hz, nblocks,
                               nsamples=NSAMPLES_FFT):
    """
    Capture averaged power spectra on both SDRs simultaneously.
    Used by calibrate_noise_diode for parallel diode-on/off measurements.
    Returns (rf, P0), (rf, P1) -- rf is identical for both since same center_hz.
    """
    results = {}

    def capture(key, sdr):
        results[key] = capture_spectrum(sdr, center_hz, nblocks)

    t0 = threading.Thread(target=capture, args=('pol0', sdr0), daemon=True)
    t1 = threading.Thread(target=capture, args=('pol1', sdr1), daemon=True)
    t0.start()
    time.sleep(0.25)   # stagger to avoid simultaneous USB calls
    t1.start()
    t0.join()
    t1.join()

    return results['pol0'], results['pol1']


# -----------------------------------------------------------------------------
# 6.  NOISE DIODE CALIBRATION  (Y-factor per polarisation)
# -----------------------------------------------------------------------------

def calibrate_noise_diode(sdr0, sdr1, noise, center_hz,
                           nblocks=NBLOCKS_CAL, nsamples=NSAMPLES_FFT):
    """
    Fire noise diode, collect spectra on both pols with diode ON and OFF.
    Returns Tsys estimate for each polarisation.

    Tsys = (T_ND * P_off) / (P_on - P_off)
    """
    print("[cal] Noise diode calibration starting...")

    # Diode OFF -- capture both pols in parallel
    noise.off()
    time.sleep(0.5)
    (_, P0_off), (_, P1_off) = capture_spectrum_parallel(sdr0, sdr1, center_hz, nblocks)

    # Diode ON -- capture both pols in parallel
    noise.on()
    time.sleep(0.5)
    (_, P0_on), (_, P1_on) = capture_spectrum_parallel(sdr0, sdr1, center_hz, nblocks)

    # Diode OFF again
    noise.off()
    time.sleep(0.3)

    # Band-centre mask (avoid edges)
    n = len(P0_off)
    mask = slice(n // 4, 3 * n // 4)

    def tsys(P_on, P_off, T_nd):
        ratio = np.nanmean(P_on[mask]) / np.nanmean(P_off[mask])
        return T_nd / (ratio - 1.0)

    tsys0 = tsys(P0_on, P0_off, T_ND_POL0)
    tsys1 = tsys(P1_on, P1_off, T_ND_POL1)

    print(f"[cal] Tsys pol0={tsys0:.1f} K   pol1={tsys1:.1f} K")
    return dict(Tsys_pol0=tsys0, Tsys_pol1=tsys1,
                P0_on=P0_on, P0_off=P0_off,
                P1_on=P1_on, P1_off=P1_off,
                center_hz=center_hz)


# -----------------------------------------------------------------------------
# 7.  BASELINE CLEANING
# -----------------------------------------------------------------------------

def clean_spectrum(diff_spec, med_kernel=None):
    """
    Passthrough -- returns the raw frequency-switched difference spectrum unchanged.

    Baseline removal is intentionally deferred to post-processing
    (analyse_hi_survey.py) where the raw ON/OFF spectra and the full
    velocity context are available.  In-observation filtering with medfilt
    was found to absorb real HI signal on ~10-20 km/s scales and is not
    applied here.  The interleaved frequency switching already provides
    a much flatter baseline than the previous block-switched approach.
    """
    baseline = np.zeros_like(diff_spec)
    cleaned  = diff_spec.copy()
    return cleaned, baseline


# -----------------------------------------------------------------------------
# 8.  VELOCITY AXIS
# -----------------------------------------------------------------------------

def velocity_axis(rf_hz, v_lsr_correction_kms=0.0):
    """
    Convert RF frequency axis to LSR velocity [km/s].
    v_lsr_correction should be added externally (from ugradio.coord.lsr_vel
    or equivalent) if needed.
    """
    v_obs = C_MS * (HI_FREQ - rf_hz) / HI_FREQ / 1e3   # km/s
    return v_obs + v_lsr_correction_kms


# -----------------------------------------------------------------------------
# 9.  FILE NAMING & SAVING
# -----------------------------------------------------------------------------

def make_filename(pointing, jd, lo_hz, data_dir, tag="obs"):
    """
    Encode metadata into filename so every file is self-describing.
    Format: HI_l<LLL.L>_b<+/-BB.B>_jd<XXXXXXXXX.XXXXX>_lo<MHZ>_<tag>.npz
    Saved into data_dir (the per-day directory for this session).
    """
    l_str  = f"{pointing['l']:07.3f}"
    b_str  = f"{pointing['b']:+07.3f}"
    jd_str = f"{jd:.5f}"
    lo_mhz = f"{lo_hz/1e6:.4f}MHz"
    fname  = f"HI_l{l_str}_b{b_str}_jd{jd_str}_lo{lo_mhz}_{tag}.npz"
    return Path(data_dir) / fname


def save_pointing(pointing, jd, lo_hz, rf0, spec0, rf1, spec1,
                  cal_data, data_dir, tag="obs", extra_meta=None):
    """
    Save both-polarisation spectra + full metadata to .npz atomically.
    data_dir must be the per-day directory for this session.
    """
    fname     = make_filename(pointing, jd, lo_hz, data_dir, tag)
    tmp_fname = fname.with_suffix(".tmp.npz")

    # Compute integration time from constants for the record
    t_block_sec     = NSAMPLES_FFT / SAMPLE_RATE          # seconds per block
    t_integ_sec     = NBLOCKS_OBS * t_block_sec * 2 * 2   # x2 switch pos x2 pols

    meta = dict(
        # -- Pointing ----------------------------------------------------------
        l_deg            = pointing['l'],
        b_deg            = pointing['b'],
        ra_deg           = pointing['ra'],
        dec_deg          = pointing['dec'],
        pointing_id      = pointing['pointing_id'],
        # -- Timing ------------------------------------------------------------
        jd_obs_end       = jd,                              # JD at end of integration
        jd_obs_start     = jd - t_integ_sec / 86400,       # approximate start JD
        unix_time_end    = (jd - 2440587.5) * 86400,
        date_utc         = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        integ_time_sec   = t_integ_sec,                     # total wall time on sky
        t_block_sec      = t_block_sec,                     # one FFT block duration
        # -- Instrument / SDR --------------------------------------------------
        lo_hz            = lo_hz,
        hi_rest_freq_hz  = HI_FREQ,
        freq_offset_hz   = FREQ_OFFSET,
        lo_on_hz         = lo_hz - FREQ_OFFSET,             # freq-switch ON position
        lo_off_hz        = lo_hz + FREQ_OFFSET,             # freq-switch OFF position
        sample_rate_hz   = SAMPLE_RATE,
        nsamples_fft     = NSAMPLES_FFT,
        nblocks_obs      = NBLOCKS_OBS,
        nblocks_cal      = NBLOCKS_CAL,
        gain_db          = GAIN,
        n_pols           = 2,
        sdr_pol0_index   = 0,
        sdr_pol1_index   = 1,
        tracker_update_sec = 20,
        # -- Site --------------------------------------------------------------
        site_name        = 'Leuschner Observatory',
        lat_deg          = LAT_DEG,
        lon_deg          = LON_DEG,
        alt_m            = ALT_M,
        # -- Calibration -------------------------------------------------------
        Tsys_pol0        = cal_data.get('Tsys_pol0', np.nan),
        Tsys_pol1        = cal_data.get('Tsys_pol1', np.nan),
        T_ND_pol0_K      = T_ND_POL0,     # noise diode increment pol 0 [K]
        T_ND_pol1_K      = T_ND_POL1,     # noise diode increment pol 1 [K]
        cal_lo_hz        = cal_data.get('center_hz', lo_hz),
        nblocks_cal_used = NBLOCKS_CAL,
        min_el_deg       = MIN_EL,
        grid_spacing_deg = GRID_SPACING_DEG,
    )
    # Keys saved explicitly in savez_compressed below -- must be removed
    # from extra_meta BEFORE meta.update() so they don't end up in **meta
    # and cause "multiple values for keyword argument" TypeError.
    EXPLICIT_KEYS = [
        'spec_on_pol0', 'spec_off_pol0', 'spec_on_pol1', 'spec_off_pol1',
        'rf_hz_on_pol0', 'rf_hz_off_pol0', 'rf_hz_on_pol1', 'rf_hz_off_pol1',
        'T_ant0_cosb', 'T_ant1_cosb',
        'on0', 'off0', 'on1', 'off1',   # legacy key names
    ]
    # Extract explicit keys from extra_meta before merging into meta
    explicit = {}
    if extra_meta:
        for k in EXPLICIT_KEYS:
            if k in extra_meta:
                explicit[k] = extra_meta.pop(k)
        meta.update(extra_meta)   # now safe -- no duplicate keys

    def _get(key, default=None):
        """Get from pre-extracted explicit keys."""
        val = explicit.get(key, explicit.get(
            {'spec_on_pol0':'on0','spec_off_pol0':'off0',
             'spec_on_pol1':'on1','spec_off_pol1':'off1'}.get(key, ''),
            default if default is not None else np.array([])))
        return val if val is not None else (default if default is not None else np.array([]))

    # Compute velocity axis for storage
    vel_kms = C_MS * (HI_FREQ - rf0) / HI_FREQ / 1e3

    np.savez_compressed(
        tmp_fname,
        # Velocity axis
        velocity_kms    = vel_kms,
        # Scaled T_ant spectra (pol0 and pol1 on ON frequency grid)
        spec_pol0       = spec0,
        spec_pol1       = spec1,
        # Raw ON spectra on their native frequency grids
        spec_on_pol0    = _get('spec_on_pol0'),
        spec_off_pol0   = _get('spec_off_pol0'),
        spec_on_pol1   = _get('spec_on_pol1'),
        spec_off_pol1   = _get('spec_off_pol1'),
        # Frequency axes for ON and OFF positions
        rf_hz_pol0      = rf0,
        rf_hz_pol1      = rf1,
        rf_hz_on_pol0   = _get('rf_hz_on_pol0',  rf0),
        rf_hz_off_pol0  = _get('rf_hz_off_pol0'),
        rf_hz_on_pol1   = _get('rf_hz_on_pol1',  rf1),
        rf_hz_off_pol1  = _get('rf_hz_off_pol1'),
        # cos(b) corrected spectra
        T_ant0_cosb     = _get('T_ant0_cosb'),
        T_ant1_cosb     = _get('T_ant1_cosb'),
        # Noise diode calibration spectra
        P0_on           = cal_data.get('P0_on',  np.array([])),
        P0_off          = cal_data.get('P0_off', np.array([])),
        P1_on           = cal_data.get('P1_on',  np.array([])),
        P1_off          = cal_data.get('P1_off', np.array([])),
        **meta
    )
    os.replace(tmp_fname, fname)
    print(f"[save] -> {fname.name}")
    return fname


# -----------------------------------------------------------------------------
# 10.  TELESCOPE TRACKING THREAD
# -----------------------------------------------------------------------------

class TelescopeTracker:
    """
    Background thread that keeps the Leuschner dish pointed at a fixed
    (RA, Dec) target, updating pointing every `update_sec` seconds.
    Call .start(pointing) to begin tracking, .stop() when done.
    """

    def __init__(self, dish, update_sec=20):
        self.dish       = dish
        self.update_sec = update_sec
        self._thread    = None
        self._stop_evt  = threading.Event()
        self.current_altaz = (np.nan, np.nan)
        self.lock       = threading.Lock()

    def start(self, ra_deg, dec_deg):
        self._stop_evt.clear()
        self._ra  = ra_deg
        self._dec = dec_deg
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="tracker"
        )
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                jd      = timing.julian_date()
                alt, az = coord.get_altaz(
                    self._ra, self._dec, jd,
                    LAT_DEG, LON_DEG, ALT_M, equinox='J2000'
                )
                soft_limit = MIN_EL - TRACK_SOFT_MARGIN
                if alt >= soft_limit:
                    with _dish_lock:
                        self.dish.point(alt, az)
                    with self.lock:
                        self.current_altaz = (alt, az)
                    if alt < MIN_EL:
                        print(f"[tracker] alt={alt:.2f}deg  az={az:.2f}deg"
                              f"  (soft tracking below MIN_EL)")
                    else:
                        print(f"[tracker] alt={alt:.2f}deg  az={az:.2f}deg")
                else:
                    print(f"[tracker] el={alt:.1f}deg below soft limit"
                          f" ({soft_limit:.1f}deg) -- holding position.")
            except Exception as e:
                print(f"[tracker] Warning: {e}")
            self._stop_evt.wait(self.update_sec)


# -----------------------------------------------------------------------------
# 11.  MAIN OBSERVATION FUNCTION  (one pointing)
# -----------------------------------------------------------------------------

def observe_pointing(pointing, dish, sdr0, sdr1, noise,
                     tracker, cal_data, data_dir, dry_run=False):
    """
    Full observing sequence for a single grid pointing:
      1. Point dish & start tracking
      2. Frequency-switch observation on both polarisations
      3. Clean spectra, compute velocity axis
      4. Save to per-day data_dir
    Returns filename or None if pointing not visible.
    """
    ra, dec = pointing['ra'], pointing['dec']

    # Single visibility check -- hard limit at MIN_EL
    jd     = timing.julian_date()
    result = pointing_is_visible(ra, dec, jd)
    if result is None:
        print(f"[obs] {pointing['pointing_id']} not visible "
              f"(el < {MIN_EL}deg) -- skipping.")
        return None
    alt0, az0 = result

    print(f"\n{'='*60}")
    print(f"[obs] Pointing: {pointing['pointing_id']}  "
          f"l={pointing['l']:.2f}deg  b={pointing['b']:+.2f}deg")
    print(f"      RA={ra:.3f}deg  Dec={dec:+.3f}deg  "
          f"alt={alt0:.1f}deg  az={az0:.1f}deg")

    if dry_run:
        print("[obs] DRY RUN -- skipping hardware commands.")
        time.sleep(0.5)
        return "DRY_RUN"

    # --- Point and begin tracking ---
    try:
        with _dish_lock:
            dish.point(alt0, az0)
    except Exception as e:
        print(f"[obs] Pointing error: {e} -- skipping.")
        return None

    tracker.start(ra, dec)
    time.sleep(3)  # settle
    jd_obs_start = timing.julian_date()   # record start before data capture

    # LSR velocity correction for this pointing and time
    try:
        v_lsr_corr = coord.lsr_vel(ra, dec, jd_obs_start)
        print(f"[obs] LSR correction: {v_lsr_corr:+.2f} km/s")
    except Exception as e:
        v_lsr_corr = 0.0
        print(f"[obs] Warning: LSR correction unavailable ({e}), using 0")

    # --- cos(b) aperture correction (computed once per pointing) -------------
    b_rad = np.radians(pointing['b'])
    cosb  = max(np.cos(b_rad), 0.01)
    print(f"[obs] b={pointing['b']:+.1f}deg  cos(b)={cosb:.3f}  "
          f"aperture correction={1/cosb:.2f}x")

    # --- Multi-window velocity coverage loop ---------------------------------
    # Observe at each velocity window to cover standard HI and HVC ranges.
    # Each window shifts the LO to place a different LSR velocity range
    # in the valid overlap bandwidth (+/- ~42 km/s per window).
    # Select velocity windows based on observing mode
    if OBSERVE_MODE == "hvc_only":
        windows_to_observe = [w for w in VELOCITY_WINDOWS if w[0] == "hvc"]
    elif OBSERVE_MODE == "std_only":
        windows_to_observe = [w for w in VELOCITY_WINDOWS if w[0] == "std"]
    else:   # "both"
        windows_to_observe = VELOCITY_WINDOWS
    print(f"[obs] Observing {len(windows_to_observe)} velocity window(s): "
          f"{[w[0] for w in windows_to_observe]}")

    fnames = []
    Tsys0  = cal_data.get('Tsys_pol0', 100.0)
    Tsys1  = cal_data.get('Tsys_pol1', 100.0)

    for win_label, v_centre_kms in windows_to_observe:

        if tracker._stop_evt.is_set():
            print(f"[obs] Quit -- stopping after windows so far")
            break

        lo_centre      = vel_to_lo(v_centre_kms)
        vel_window_kms = (FREQ_OFFSET / HI_FREQ) * C_MS / 1e3
        is_hvc         = abs(v_centre_kms) >= 90.0

        print(f"[obs] Window '{win_label}': v_centre={v_centre_kms:+.0f} km/s  "
              f"range=[{v_centre_kms-vel_window_kms:+.0f}, "
              f"{v_centre_kms+vel_window_kms:+.0f}] km/s  "
              f"LO={lo_centre/1e6:.4f} MHz")

        rf_on0, on0, rf_off0, off0, rf_on1, on1, rf_off1, off1 = freq_switch_pair_parallel(
            sdr0, sdr1, lo_centre, FREQ_OFFSET, NBLOCKS_OBS,
            quit_event=tracker._stop_evt
        )
        rf0 = rf_on0   # primary freq axis (ON position, pol0)

        with tracker.lock:
            mid_altaz = tracker.current_altaz
        jd_win = timing.julian_date()

        # T_ant scaling using OFF spectrum mean for normalisation
        # Both ON and OFF are on their own native frequency grids --
        # alignment is done in post-processing.
        n    = len(off0)
        band = slice(n // 4, 3 * n // 4)
        P0_off_mean = np.nanmean(off0[band])
        P1_off_mean = np.nanmean(off1[band])
        if not np.isfinite(P0_off_mean) or P0_off_mean == 0: P0_off_mean = 1.0
        if not np.isfinite(P1_off_mean) or P1_off_mean == 0: P1_off_mean = 1.0

        # Scaled ON spectrum as primary data product
        # T_ant = Tsys * ON / mean(OFF)  -- baseline removal done in post
        T_ant0      = Tsys0 * on0 / P0_off_mean
        T_ant1      = Tsys1 * on1 / P1_off_mean
        T_ant0_cosb = T_ant0 / cosb
        T_ant1_cosb = T_ant1 / cosb

        vel_kms = velocity_axis(rf_on0, v_lsr_correction_kms=v_lsr_corr)

        extra = dict(
            alt_deg_mid          = mid_altaz[0],
            az_deg_mid           = mid_altaz[1],
            alt_deg_start        = alt0,
            az_deg_start         = az0,
            jd_obs_start         = jd_obs_start,
            noise_diode          = "off",
            v_lsr_correction_kms = v_lsr_corr,
            pll_settle_sec       = PLL_SETTLE_SEC,
            win_label            = win_label,
            v_centre_kms         = v_centre_kms,
            lo_centre_hz         = lo_centre,
            lo_on_hz             = lo_centre - FREQ_OFFSET,
            lo_off_hz            = lo_centre + FREQ_OFFSET,
            vel_window_kms       = vel_window_kms,
            is_hvc_window        = is_hvc,
            cosb                 = cosb,
            cosb_correction      = 1.0 / cosb,
            # spec_on/off, rf grids, T_ant_cosb are passed directly to
            # save_pointing as positional/keyword args -- not duplicated here
            on0 = on0, off0 = off0, on1 = on1, off1 = off1,
            rf_hz_on_pol0 = rf_on0, rf_hz_off_pol0 = rf_off0,
            rf_hz_on_pol1 = rf_on1, rf_hz_off_pol1 = rf_off1,
            T_ant0_cosb = T_ant0_cosb, T_ant1_cosb = T_ant1_cosb,
        )

        win_fname = save_pointing(
            pointing, jd_win, lo_centre,
            rf_on0, T_ant0, rf_on1, T_ant1,
            cal_data, data_dir=data_dir,
            tag=f"obs_{win_label}", extra_meta=extra
        )
        fnames.append(win_fname)
        print(f"[obs] '{win_label}' saved: {win_fname.name}")

    tracker.stop()
    jd_obs = timing.julian_date()
    fname  = fnames[0] if fnames else None


    # --- Quick-look SNR estimate printed to terminal ---
    valid = np.isfinite(T_ant0) & (np.abs(T_ant0) > 0)
    if valid.sum() > 200:
        n      = valid.sum()
        trim   = n // 10
        idx    = np.where(valid)[0]
        v_q    = vel_kms[idx[trim:-trim]]
        s_q    = ((T_ant0 + T_ant1) / 2)[idx[trim:-trim]]
        # Estimate noise from outer 20% of band (line-free)
        edge   = len(v_q) // 5
        rms_q  = np.std(np.concatenate([s_q[:edge], s_q[-edge:]]))
        peak_T = np.max(s_q)
        peak_v = v_q[np.argmax(s_q)]
        snr_q  = peak_T / rms_q if rms_q > 0 else 0
        print(f"[obs] Quick-look: peak={peak_T:.2f}K at v={peak_v:.1f}km/s  "
              f"rms={rms_q:.2f}K  SNR={snr_q:.1f}  "
              f"(LSR corr={v_lsr_corr:+.1f}km/s)")
    print(f"[obs] Done. Tsys0={Tsys0:.1f}K  Tsys1={Tsys1:.1f}K")
    return fname


# -----------------------------------------------------------------------------
# 12.  ELEVATION-PRIORITY SCHEDULER
# -----------------------------------------------------------------------------

# Minimum clearance above the local obstruction profile required to observe.
# Pointings with less than this many degrees of clearance above the nearest
# hill/treeline are deferred even if technically unobstructed.
# Increase this if RFI or diffraction from nearby terrain is a problem.
MIN_CLEARANCE_DEG = 2.0


def sort_by_elevation_priority(pending, jd):
    """
    Sort pending pointings by scheduling priority accounting for local terrain.

    Priority logic
    --------------
    Visible pointings are split into two tiers:

    Tier 1 -- LOW but RISING (clearance < 20deg above local horizon):
        Observed first because they have a short window before setting.
        Sorted ascending by clearance (least margin first = most urgent).

    Tier 2 -- HIGH and SAFE (clearance >= 20deg above local horizon):
        Observed after tier-1. These are well clear of terrain and will
        remain visible for hours, so we let the urgent low ones go first.
        Sorted descending by clearance (highest first = best data quality,
        least atmospheric absorption, lowest Tsys).

    Obstructed / not yet risen:
        Placed at the end, sorted by declination (higher dec rises sooner
        at this latitude).

    Pointings with clearance < MIN_CLEARANCE_DEG above the local horizon
    profile are treated as obstructed regardless of mathematical visibility.

    Returns a new sorted list of (pointing, alt_deg) tuples.
    """
    RISING_THRESHOLD_DEG = 45.0   # deg clearance above local horizon
    # With MIN_EL=35deg all visible pointings have >30deg clearance,
    # so threshold should be well above 20deg to meaningfully split tiers.
    # Pointings <45deg clearance are more urgent (setting sooner);
    # pointings >45deg clearance are safe and observed highest-first.

    tier1 = []   # low, urgent -- observe first
    tier2 = []   # high, safe  -- observe after tier1, highest first
    blocked = [] # obstructed or not yet risen

    for pt in pending:
        result = pointing_is_visible(pt['ra'], pt['dec'], jd)
        if result is not None:
            alt, az = result
            clearance = elevation_above_local_horizon(alt, az)
            if clearance < MIN_CLEARANCE_DEG:
                # Too close to terrain -- defer
                blocked.append((pt, alt, az, clearance))
            elif clearance < RISING_THRESHOLD_DEG:
                tier1.append((pt, alt, az, clearance))
            else:
                tier2.append((pt, alt, az, clearance))
        else:
            blocked.append((pt, -999.0, 0.0, -999.0))

    # Tier 1: lowest clearance first (most urgent)
    tier1.sort(key=lambda x: x[3])
    # Tier 2: highest clearance first (best quality)
    tier2.sort(key=lambda x: -x[3])
    # Blocked: sort by dec so higher-dec (rises sooner) comes first
    blocked.sort(key=lambda x: -x[0]['dec'])

    print(f"[scheduler] Tier1 (urgent, low)={len(tier1)}  "
          f"Tier2 (safe, high)={len(tier2)}  "
          f"Blocked/not risen={len(blocked)}")

    if tier1:
        lo = tier1[0][3]
        hi = tier1[-1][3]
        print(f"[scheduler]   Tier1 clearance range: {lo:.1f}deg - {hi:.1f}deg above horizon")
    if tier2:
        lo = tier2[-1][3]
        hi = tier2[0][3]
        print(f"[scheduler]   Tier2 clearance range: {lo:.1f}deg - {hi:.1f}deg above horizon")

    ordered = tier1 + tier2 + blocked
    return [(x[0], x[1]) for x in ordered]


def run_survey(resume=True, dry_run=False, resort_every=30):
    """
    Main survey loop with elevation-priority scheduling and per-day directories.

    Scheduling logic
    ----------------
    At the start of the session (and every `resort_every` successful
    observations) the pending list is re-sorted by current elevation so that
    the pointing nearest the horizon is always targeted first.  This ensures
    no pointing is lost because it set while we were observing something
    that could have waited.

    Directory layout
    ----------------
    data/
      2025-06-01/
        progress_2025-06-01.json   <- pointings done on this night
        HI_l*.npz  ...
      2025-06-02/
        progress_2025-06-02.json
        HI_l*.npz  ...
    survey_progress.json           <- union across all nights (global)

    Recovery
    --------
    If a night's data is corrupted, delete data/<date>/ and run:
        python3 -c "from observe_hi_survey import rebuild_global_progress; rebuild_global_progress()"
    This rescans all remaining day folders and rebuilds the global progress
    file, so those pointings will be re-queued automatically.
    """
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    data_dir = get_day_dir(date_str)
    print(f"[survey] Data directory for this session: {data_dir}")

    grid           = make_survey_grid()
    global_done    = load_global_progress() if resume else set()
    day_done       = load_day_progress(date_str) if resume else set()

    pending_all = [p for p in grid if p['pointing_id'] not in global_done]
    print(f"[survey] {len(pending_all)} pointings remaining "
          f"({len(global_done)} already done globally, "
          f"{len(day_done)} done today).")

    if not pending_all:
        print("[survey] Survey complete!")
        return

    # -- Hardware init (inside try/finally so SDRs always get closed) ---------
    # Initialise to None so finally block can safely check before closing
    dish  = None
    noise = None
    sdr0  = None
    sdr1  = None

    # -- Pre-session radiometer summary --------------------------------------
    t_block     = NSAMPLES_FFT / SAMPLE_RATE
    t_int       = NBLOCKS_OBS * t_block
    delta_nu    = SAMPLE_RATE / NSAMPLES_FFT
    N_smooth    = max(1, int(1e3 / (SAMPLE_RATE / NSAMPLES_FFT / HI_FREQ * C_MS / 1e3)))
    Tsys_est    = 100.0   # K -- rough estimate, updated after calibration
    sigma_chan  = Tsys_est / (delta_nu * t_int) ** 0.5
    sigma_1kms  = sigma_chan / N_smooth ** 0.5
    win_labels  = ([w[0] for w in VELOCITY_WINDOWS if w[0] == "hvc"]
                   if OBSERVE_MODE == "hvc_only" else
                   [w[0] for w in VELOCITY_WINDOWS if w[0] == "std"]
                   if OBSERVE_MODE == "std_only" else
                   [w[0] for w in VELOCITY_WINDOWS])
    t_per_point = t_int * len(win_labels) * 2   # x2 for ON+OFF
    t_total_hr  = 1281 * (t_per_point + 30) / 3600   # +30s slew overhead

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  HI 21-cm Survey -- Session Parameters")
    print(f"{sep}")
    print(f"  Mode           : {OBSERVE_MODE}")
    print(f"  Windows        : {win_labels}")
    print(f"  NBLOCKS_OBS    : {NBLOCKS_OBS}  ({t_int:.0f}s per LO position)")
    print(f"  Time/pointing  : ~{t_per_point/60:.1f} min")
    print(f"  Full survey    : ~{t_total_hr:.0f} hours  (~{t_total_hr/6:.0f} nights at 6hr/night)")
    print(f"\n  Radiometer equation (Tsys~{Tsys_est:.0f}K, smooth to 1 km/s):")
    print(f"    sigma/channel  = {sigma_chan:.2f} K")
    print(f"    sigma (1 km/s) = {sigma_1kms:.2f} K  (smoothed over ~{N_smooth} channels)")
    for T_hvc in [3.0, 1.0, 0.5]:
        snr = T_hvc / sigma_1kms
        n_visits = max(1, int((5.0 / snr) ** 2 + 0.5))
        print(f"    SNR({T_hvc:.1f}K HVC)  = {snr:.1f} sigma/visit"
              f"  -> need ~{n_visits} visit(s) for 5-sigma")
    print(f"{sep}\n")

    print("[init] Connecting to Leuschner dish...")
    dish  = leusch.LeuschTelescope()
    noise = leusch.LeuschNoise()
    noise.off()

    print("[init] Opening SDRs...")
    sdr0 = ugradio.sdr.SDR(device_index=0, direct=False,
                           center_freq=HI_FREQ,
                           sample_rate=SAMPLE_RATE,
                           gain=GAIN)
    sdr1 = ugradio.sdr.SDR(device_index=1, direct=False,
                           center_freq=HI_FREQ,
                           sample_rate=SAMPLE_RATE,
                           gain=GAIN)
    print(f"[init] SDR0 gain={sdr0.get_gain():.1f}dB  "
          f"sample_rate={sdr0.get_sample_rate()/1e6:.2f}MHz")
    print(f"[init] SDR1 gain={sdr1.get_gain():.1f}dB  "
          f"sample_rate={sdr1.get_sample_rate()/1e6:.2f}MHz")

    tracker = TelescopeTracker(dish, update_sec=20)

    # Verify and correct gain before calibration.
    # RTL-SDR snaps to nearest valid gain step -- find the closest step
    # to GAIN that is still in the linear (non-saturating) regime.
    if not dry_run:
        for label, sdr in [('SDR0', sdr0), ('SDR1', sdr1)]:
            valid = sorted(sdr.valid_gains_db)
            # Pick closest valid gain to GAIN
            best = min(valid, key=lambda g: abs(g - GAIN))
            print(f"[init] {label} valid gains: {valid}")
            print(f"[init] {label} requesting gain={GAIN}dB  "
                  f"-> nearest valid={best:.1f}dB")
            sdr.set_gain(best)
            time.sleep(0.2)
            actual_gain = sdr.get_gain()
            actual_rate = sdr.get_sample_rate()
            print(f"[init] {label} confirmed: gain={actual_gain:.1f}dB  "
                  f"rate={actual_rate/1e6:.2f}MHz")

    # Initial calibration
    if not dry_run:
        cal_data = calibrate_noise_diode(sdr0, sdr1, noise, HI_FREQ)
    else:
        cal_data = dict(Tsys_pol0=100.0, Tsys_pol1=100.0)


    # -- Command menu thread ---------------------------------------------------
    # A shared event lets the menu thread signal a clean stop after the
    # current pointing finishes -- we never interrupt a capture mid-block.
    quit_event = threading.Event()

    def command_menu():
        """
        Background thread accepting commands while observation runs.
        Uses sys.stdin.readline() instead of input() to avoid screen/tmux
        sending terminal escape sequences into the input buffer.

        Commands
        --------
        status  : print current progress and next scheduled target
        quit    : finish the current pointing then stop cleanly
        Ctrl-C  : immediate stop (KeyboardInterrupt in main thread)
        """
        print("\n[menu] Command menu ready.  Commands: status / quit")
        while not quit_event.is_set():
            try:
                # Use sys.stdin.readline() -- more robust than input() in
                # screen/tmux sessions which can inject escape sequences
                line = sys.stdin.readline()
                if not line:
                    # EOF -- stdin closed, run silently without menu
                    time.sleep(1)
                    continue
                # Strip ANSI escape sequences and whitespace before parsing
                import re
                cmd = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip().lower()
                if not cmd:
                    continue
            except Exception:
                time.sleep(0.5)
                continue

            if cmd == "quit":
                print("[menu] Quit received -- will stop after current pointing.")
                quit_event.set()
                break
            elif cmd == "status":
                n_rem = len(pending_iter) if "pending_iter" in dir() else "?"
                print(f"[menu] Global done : {len(global_done)}/{len(grid)}"
                      f"  ({100*len(global_done)/max(len(grid),1):.1f}%)")
                print(f"[menu] Today done  : {len(day_done)}")
                print(f"[menu] Remaining   : {n_rem} pointings this session")
            elif cmd in ("", "\n"):
                pass
            else:
                # Silently ignore escape sequences and other control chars
                if not cmd.startswith("\x1b") and len(cmd) < 20:
                    print(f"[menu] Unknown command '{cmd}'.  Try: status / quit")
        print("[menu] Command menu exiting.")

    menu_thread = threading.Thread(target=command_menu, daemon=True, name="menu")
    menu_thread.start()

    # -- Observation loop -----------------------------------------------------
    obs_count   = 0
    skip_count  = 0

    # Initial elevation-priority sort
    jd_now  = timing.julian_date()
    pending = sort_by_elevation_priority(pending_all, jd_now)

    # pending is now a list of (pointing_dict, current_alt) tuples
    pending_iter = list(pending)   # we'll consume this and refresh periodically

    try:
        while pending_iter and not quit_event.is_set():
            # Re-sort every resort_every observations to adapt to sky rotation
            if obs_count > 0 and obs_count % resort_every == 0:
                print(f"\n[scheduler] Re-sorting by elevation "
                      f"(obs #{obs_count}) ...")
                remaining_pts = [x[0] for x in pending_iter]
                jd_now  = timing.julian_date()
                pending_iter = sort_by_elevation_priority(remaining_pts, jd_now)

            # Recalibrate every CAL_EVERY_N successful observations
            if obs_count > 0 and obs_count % CAL_EVERY_N == 0:
                print(f"\n[survey] -- Recalibrating (obs #{obs_count}) --")
                if not dry_run:
                    cal_data = calibrate_noise_diode(sdr0, sdr1, noise, HI_FREQ)

            pointing, sched_alt = pending_iter.pop(0)
            pid = pointing['pointing_id']

            print(f"\n[scheduler] Next target: {pid}  "
                  f"(scheduled el={sched_alt:.1f}deg)")

            try:
                fname = observe_pointing(
                    pointing, dish, sdr0, sdr1, noise,
                    tracker, cal_data, data_dir=data_dir, dry_run=dry_run
                )
            except Exception as obs_err:
                # Log the error but keep the survey running
                print(f"[survey] ERROR observing {pid}: {obs_err}")
                import traceback
                traceback.print_exc()
                # Re-queue this pointing and try to recover the SDRs
                skip_count += 1
                pending_iter.append((pointing, sched_alt))
                # Give hardware time to recover before next attempt
                print("[survey] Waiting 30s for hardware recovery...")
                time.sleep(30)
                fname = None

            if fname is not None and fname != "DRY_RUN":
                global_done.add(pid)
                day_done.add(pid)
                save_progress(global_done, day_done, date_str)
                obs_count += 1
                skip_count = 0
            elif fname == "DRY_RUN":
                obs_count += 1
            else:
                # Pointing was not visible or failed -- put it at the back
                skip_count += 1
                if (pointing, sched_alt) not in pending_iter:
                    pending_iter.append((pointing, sched_alt))
                if skip_count >= len(pending_iter):
                    print("[survey] All remaining pointings currently below "
                          "horizon -- waiting 5 minutes before retrying.")
                    time.sleep(300)
                    skip_count = 0
                    jd_now = timing.julian_date()
                    pending_iter = sort_by_elevation_priority(
                        [x[0] for x in pending_iter], jd_now
                    )

            n_total = len(grid)
            print(f"[survey] Progress: {len(global_done)}/{n_total} "
                  f"({100*len(global_done)/n_total:.1f}%)  |  "
                  f"today: {len(day_done)}")

    except KeyboardInterrupt:
        print("\n[survey] Interrupted by user (Ctrl-C). Progress saved.")

    finally:
        # Signal menu thread to stop and wait briefly
        quit_event.set()
        menu_thread.join(timeout=2)

        tracker.stop()

        if not pending_iter:
            print("[survey] Clean stop -- all pending pointings completed.")
        else:
            print(f"[survey] Stopping with {len(pending_iter)} pointings still remaining.")

        print("[survey] Stowing dish...")
        if not dry_run and dish is not None:
            try:
                dish.stow()
            except Exception as e:
                print(f"[survey] Stow warning: {e}")

        # Always close SDRs if they were opened -- prevents LIBUSB_ERROR_BUSY
        # on the next run if this session crashed
        for label, sdr in [("SDR0", sdr0), ("SDR1", sdr1)]:
            if sdr is not None:
                try:
                    sdr.close()
                    print(f"[survey] {label} closed.")
                except Exception as e:
                    print(f"[survey] {label} close warning: {e}")

        save_progress(global_done, day_done, date_str)
        print(f"[survey] Session complete -- "
              f"{obs_count} pointings observed this session  |  "
              f"data in {data_dir}")


# -----------------------------------------------------------------------------
# 13.  ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HI 21-cm Galactic Survey -- Leuschner Dish"
    )
    parser.add_argument('--plan',    action='store_true',
                        help='Print visibility windows for the next 12 h and exit')
    parser.add_argument('--resume',  action='store_true', default=True,
                        help='Resume from progress file (default: True)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Ignore progress file and start fresh')
    parser.add_argument('--dry-run', action='store_true',
                        help='Step through grid without moving telescope or capturing data')
    parser.add_argument('--mode', choices=['hvc_only', 'both', 'std_only'],
                        default=None,
                        help='Observing mode: hvc_only (default), both, or std_only. '
                             'Overrides OBSERVE_MODE constant in script.')
    parser.add_argument('--resort-every', type=int, default=30, metavar='N',
                        help='Re-sort by elevation every N successful observations (default 10)')
    parser.add_argument('--rebuild-progress', action='store_true',
                        help='Rebuild global progress from per-day files and exit '
                             '(use after deleting a corrupted day folder)')
    args = parser.parse_args()

    if args.rebuild_progress:
        rebuild_global_progress()
        return

    grid = make_survey_grid()

    if args.plan:
        print_visibility_windows(grid, hours=14, step_min=5)
        return

    # Override OBSERVE_MODE if --mode was passed on command line
    if args.mode is not None:
        import observe_hi_survey as _self
        _self.OBSERVE_MODE = args.mode
        _self.OBSERVE_HVC  = args.mode in ("hvc_only", "both")

    run_survey(resume=args.resume, dry_run=args.dry_run,
               resort_every=args.resort_every)


if __name__ == "__main__":
    main()
