import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
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
    if not (6 < alt < 174 and 88 < az < 300):
        print(f"WARNING: Out of bounds (alt={alt:.2f}, az={az:.2f})")
        return False
    ifm.point(alt, az)
    return True

# --- Live plotting setup ---
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])

ax.set_title("Live Fringe (Channel 500)")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")

buffer_size = 200
ydata = deque(maxlen=buffer_size)

# --- Initial pointing ---
alt, az = get_sun_altaz()
print(f"Initial pointing: alt={alt:.2f}, az={az:.2f}")

if not safe_point(alt, az):
    raise RuntimeError("Sun not observable!")

print("Starting observation...")

last_point_time = time.time()

try:
    while True:
        # --- Update pointing every 30 seconds (FIXED) ---
        if time.time() - last_point_time > 30:
            alt, az = get_sun_altaz()
            safe_point(alt, az)
            last_point_time = time.time()

        # --- Read SNAP data ---
        d = spec.read_data(acc_cnt=acc_cnt_prev)
        acc_cnt_prev = d['acc_cnt']

        # --- Save full spectrum ---
        data.append({
            'time': time.time(),
            'acc_cnt': d['acc_cnt'],
            'corr_real': d['corr00'].real,
            'corr_imag': d['corr00'].imag
        })

        # --- Live plot (single channel) ---
        value = d['corr00'].real[500]
        ydata.append(value)

        if len(ydata) % 5 == 0:
            line.set_xdata(range(len(ydata)))
            line.set_ydata(ydata)

            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.01)

        print(f"acc_cnt: {d['acc_cnt']}")

except KeyboardInterrupt:
    print("Stopping observation...")

# --- Save data ---
np.save(filename, data)
print(f"Saved to {filename}")

# --- STOW DISHES ---
ifm.stow()
print("Telescopes stowed.")
