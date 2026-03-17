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

# --- Storage ---
data = []
acc_cnt_prev = None

def get_sun_altaz(jd):
    ra, dec = ugradio.coord.sunpos(jd)
    alt, az = ugradio.coord.get_altaz(ra, dec, jd, LAT, LON, ALT)
    return ra, dec, alt, az

def safe_point(alt, az):
    if not (6 < alt < 174 and 88 < az < 300):
        print(f"WARNING: Out of bounds (alt={alt:.2f}, az={az:.2f})")
        return False
    ifm.point(alt, az)
    return True

# --- Live plotting ---
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("Live Fringe (Channel 500)")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")

buffer_size = 200
ydata = deque(maxlen=buffer_size)

# --- Initial pointing ---
jd = ugradio.timing.julian_date()
ra, dec, alt, az = get_sun_altaz(jd)

print(f"Initial pointing: alt={alt:.2f}, az={az:.2f}")

if not safe_point(alt, az):
    raise RuntimeError("Sun not observable!")

last_point_time = time.time()

print("Starting observation...")

try:
    while True:
        # --- Time ---
        t_unix = time.time()
        jd = ugradio.timing.julian_date()

        # --- Sun position (commanded) ---
        ra, dec, alt_cmd, az_cmd = get_sun_altaz(jd)

        # --- Repoint every 30 seconds ---
        if t_unix - last_point_time > 30:
            print(f"Repointing: alt={alt_cmd:.2f}, az={az_cmd:.2f}")
            safe_point(alt_cmd, az_cmd)
            last_point_time = t_unix

        # --- Actual telescope pointing (ENCODERS) ---
        try:
            pointing = ifm.get_pointing()
            alt_w, az_w = pointing['ant_w']
            alt_e, az_e = pointing['ant_e']

            alt_meas = (alt_w + alt_e) / 2
            az_meas = (az_w + az_e) / 2
        except:
            alt_meas = None
            az_meas = None

        # --- Read SNAP data ---
        d = spec.read_data(acc_cnt=acc_cnt_prev)
        acc_cnt_prev = d['acc_cnt']

        # --- Save EVERYTHING ---
        data.append({
            'time_unix': t_unix,
            'jd': jd,
            'acc_cnt': d['acc_cnt'],

            # sky position
            'ra': ra,
            'dec': dec,

            # commanded pointing
            'alt_cmd': alt_cmd,
            'az_cmd': az_cmd,

            # measured pointing (encoders)
            'alt_meas': alt_meas,
            'az_meas': az_meas,

            # visibilities
            'corr_real': d['corr00'].real,
            'corr_imag': d['corr00'].imag
        })

        # --- Live plot ---
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

# --- STOW ---
ifm.stow()
print("Telescopes stowed.")
