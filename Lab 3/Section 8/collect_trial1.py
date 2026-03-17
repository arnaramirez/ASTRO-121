import time
import numpy as np
import ugradio
from snap_spec.snap import UGRadioSnap

# --- Observatory location ---
LAT = 37.873199
LON = -122.257063
ALT = 120.0

# --- Initialize interferometer ---
ifm = ugradio.interf.Interferometer()

# --- Initialize SNAP ---
spec = UGRadioSnap(host='localhost', is_discover=True)
spec.initialize(mode='corr', sample_rate=500)

# --- Output file ---
filename = "sun_observation.npy"

# --- Storage buffers ---
data = []
acc_cnt_prev = None

def get_sun_altaz():
    jd = ugradio.timing.julian_date()
    ra, dec = ugradio.coord.sunpos(jd)
    alt, az = ugradio.coord.get_altaz(ra, dec, jd, LAT, LON, ALT)
    return alt, az

def safe_point(alt, az):
    """Ensure pointing is within allowed bounds."""
    if not (6 < alt < 174 and 88 < az < 300):
        print(f"WARNING: Target out of bounds (alt={alt}, az={az})")
        return False
    ifm.point(alt, az)
    return True

# --- Initial pointing ---
alt, az = get_sun_altaz()
print(f"Initial pointing: alt={alt:.2f}, az={az:.2f}")

if not safe_point(alt, az):
    raise RuntimeError("Sun not observable right now!")

print("Starting observation...")

try:
    while True:
        # --- Update pointing every ~30 sec ---
        if int(time.time()) % 30 == 0:
            alt, az = get_sun_altaz()
            safe_point(alt, az)

        # --- Read SNAP data ---
        d = spec.read_data(acc_cnt=acc_cnt_prev)
        acc_cnt_prev = d['acc_cnt']

        # --- Store FULL visibility spectrum ---
        data.append({
            'time': time.time(),
            'acc_cnt': d['acc_cnt'],
            'corr_real': d['corr00'].real,
            'corr_imag': d['corr00'].imag
        })

        print(f"acc_cnt: {d['acc_cnt']}")

except KeyboardInterrupt:
    print("Stopping observation...")

# --- Save data ---
np.save(filename, data)
print(f"Saved to {filename}")

# --- STOW DISHES (VERY IMPORTANT) ---
ifm.stow()
print("Telescopes stowed.")
