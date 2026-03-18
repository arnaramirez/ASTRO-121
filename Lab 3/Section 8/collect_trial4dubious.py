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

# --- Rolling buffers for live plotting ---
buffer_size = 200
ydata = deque(maxlen=buffer_size)
xdata = deque(maxlen=buffer_size)

plt.ion()  # interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("Live Fringe (Single Channel)")
ax.set_xlabel("Time index")
ax.set_ylabel("Amplitude")

# --- Storage ---
data = []
acc_cnt_prev = None

# --- Functions ---
def get_sun_altaz():
    jd = ugradio.timing.julian_date()
    ra, dec = ugradio.coord.sunpos(jd)
    alt, az = ugradio.coord.get_altaz(ra, dec, jd, LAT, LON, ALT)
    return alt, az, ra, dec, jd

def safe_point(alt, az):
    """Ensure pointing is within allowed bounds and point telescopes."""
    if not (6 < alt < 174 and 88 < az < 300):
        print(f"WARNING: Target out of bounds (alt={alt:.2f}, az={az:.2f})")
        return False
    ifm.point(alt, az)
    return True

# --- Initial pointing ---
alt, az, ra, dec, jd = get_sun_altaz()
print(f"Initial pointing: alt={alt:.2f}, az={az:.2f}")
if not safe_point(alt, az):
    raise RuntimeError("Sun not observable right now!")

print("Starting observation...")

try:
    while True:
        # --- Update pointing every ~30 sec ---
        if int(time.time()) % 30 == 0:
            alt, az, ra, dec, jd = get_sun_altaz()
            safe_point(alt, az)

        # --- Read SNAP data ---
        d = spec.read_data(acc_cnt=acc_cnt_prev)
        acc_cnt_prev = d['acc_cnt']

        # --- Pick ONE channel to save and plot (e.g., 500) ---
        channel_index = 500
        value_real = d['corr00'].real[channel_index]
        value_imag = d['corr00'].imag[channel_index]

        # --- Save metadata + single-channel value ---
        t_unix = time.time()
        alt_cmd, az_cmd = ifm.get_pointing()['ant_w']  # commanded pointing
        alt_meas, az_meas = alt, az  # actual sun alt/az at measurement
        data_point = {
            'time_unix': t_unix,
            'jd': jd,
            'acc_cnt': d['acc_cnt'],
            'ra': ra,
            'dec': dec,
            'alt_cmd': alt_cmd,
            'az_cmd': az_cmd,
            'alt_meas': alt_meas,
            'az_meas': az_meas,
            'value_real': value_real,
            'value_imag': value_imag
        }
        data.append(data_point)

        # --- Update rolling plot ---
        ydata.append(value_real)
        xdata.append(len(xdata))
        if len(ydata) % 5 == 0:  # update plot every 5 samples
            line.set_xdata(range(len(ydata)))
            line.set_ydata(ydata)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        # --- Incremental save to disk every 20 samples ---
        if len(data) % 20 == 0:
            np.save(filename, data)

        print(f"acc_cnt: {d['acc_cnt']}, channel[{channel_index}] = {value_real:.3f}")

except KeyboardInterrupt:
    print("Stopping observation...")

# --- Save all remaining data ---
np.save(filename, data)
print(f"Saved to {filename}")

# --- Stow telescopes ---
ifm.stow()
print("Telescopes stowed.")
