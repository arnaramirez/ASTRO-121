"""
waterfall_watcher.py
--------------------
Watches for sun_data_*.npz files and displays a live-updating waterfall
plot of the visibility spectra (amplitude vs frequency vs time).

Usage (in a separate terminal, while observe_sun.py is running):
    python3 waterfall_watcher.py

Requires:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import time
from pathlib import Path
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = f"data/{today}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_SEC     = 15
SNAP_SRATE   = 500e6
N_CHANNELS   = 1024
FMIN_GHZ = 1.415
FMAX_GHZ = 1.665
VMIN_DB      = None
VMAX_DB      = None

# ---------------------------------------------------------------------------

def get_partial_files():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "sun_data_*.npz")))
    return [f for f in files if "master" not in os.path.basename(f)]

def load_all_data():
    files = get_partial_files()
    if not files:
        return None, None

    all_vis, all_times = [], []
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            all_vis.append(d["vis"])
            all_times.append(d["times"])
        except Exception as e:
            print(f"[waterfall] Could not read {f}: {e} — skipping.")

    if not all_vis:
        return None, None

    vis   = np.concatenate(all_vis,   axis=0)
    times = np.concatenate(all_times, axis=0)

    order = np.argsort(times)
    return vis[order], times[order]

def vis_to_db(vis):
    amp = np.abs(vis)
    amp = np.where(amp == 0, 1e-10, amp)
    return 20 * np.log10(amp)

def times_to_minutes(times):
    return (times - times[0]) * 24 * 60

def make_freq_axis():
    return np.linspace(FMIN_GHZ, FMAX_GHZ, N_CHANNELS)

# ---------------------------------------------------------------------------

def main():
    print(f"[waterfall] Watching directory: {os.path.abspath(OUTPUT_DIR)}")
    print("[waterfall] Starting live waterfall watcher...")

    plt.ion()

    # --- NOW 3 PANELS: amplitude, phase, spectrum ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [3, 3, 1]})

    fig.suptitle("X-Band Interferometer — Live Waterfall", fontsize=13)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")

    ax_amp, ax_phase, ax_spec = axes
    freq_axis = make_freq_axis()

    # --- Initialize colorbars ---
    im_amp = ax_amp.imshow(np.zeros((2, N_CHANNELS)), aspect="auto", cmap="brg")
    cbar_amp = fig.colorbar(im_amp, ax=ax_amp, pad=0.02)
    cbar_amp.set_label("Amplitude (dB)", color="white")
    cbar_amp.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_amp.ax.yaxis.get_ticklabels(), color="white")

    im_phase = ax_phase.imshow(np.zeros((2, N_CHANNELS)), aspect="auto", cmap="brg")
    cbar_phase = fig.colorbar(im_phase, ax=ax_phase, pad=0.02)
    cbar_phase.set_label("Phase (rad)", color="white")
    cbar_phase.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_phase.ax.yaxis.get_ticklabels(), color="white")

    while True:
        vis, times = load_all_data()

        if vis is None or vis.shape[0] < 2:
            print("[waterfall] Waiting for data...")
            plt.pause(POLL_SEC)
            continue

        db = vis_to_db(vis)
        phase = np.angle(vis)

        t_min = times_to_minutes(times)

        # ------------------ AMPLITUDE ------------------
        ax_amp.cla()
        ax_amp.set_facecolor("#0f0f0f")

        vmin = VMIN_DB if VMIN_DB else np.percentile(db, 2)
        vmax = VMAX_DB if VMAX_DB else np.percentile(db, 98)

        im_amp = ax_amp.imshow(
            db,
            aspect="auto",
            origin="lower",
            extent=[freq_axis[0], freq_axis[-1], t_min[0], t_min[-1]],
            cmap="brg",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax_amp.set_title("Amplitude", color="white")
        ax_amp.set_xlabel("Frequency (GHz)", color="white")
        ax_amp.set_ylabel("Time (min)", color="white")
        ax_amp.tick_params(colors="white")
        for spine in ax_amp.spines.values():
            spine.set_edgecolor("white")

        im_amp.set_clim(vmin, vmax)
        cbar_amp.update_normal(im_amp)

        # ------------------ PHASE ------------------
        ax_phase.cla()
        ax_phase.set_facecolor("#0f0f0f")

        im_phase = ax_phase.imshow(
            phase,
            aspect="auto",
            origin="lower",
            extent=[freq_axis[0], freq_axis[-1], t_min[0], t_min[-1]],
            cmap="brg",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="nearest",
        )

        ax_phase.set_title("Phase", color="white")
        ax_phase.set_xlabel("Frequency (GHz)", color="white")
        ax_phase.tick_params(colors="white")
        for spine in ax_phase.spines.values():
            spine.set_edgecolor("white")

        cbar_phase.update_normal(im_phase)

        # ------------------ SPECTRUM ------------------
        ax_spec.cla()
        ax_spec.set_facecolor("#0f0f0f")
        ax_spec.plot(db[-1], freq_axis, color="#ff6b35", linewidth=0.8)

        ax_spec.set_xlabel("Amplitude (dB)", color="white")
        ax_spec.set_title("Latest Spectrum", color="white", fontsize=9)
        ax_spec.tick_params(colors="white")
        ax_spec.set_ylim(freq_axis[0], freq_axis[-1])

        for spine in ax_spec.spines.values():
            spine.set_edgecolor("white")

        # ------------------ TITLE ------------------
        n_int = vis.shape[0]
        elapsed = t_min[-1]

        fig.suptitle(
            f"X-Band Interferometer — Live Waterfall  |  "
            f"{n_int} integrations  |  {elapsed:.1f} min elapsed",
            color="white", fontsize=11
        )

        plt.tight_layout()
        plt.pause(POLL_SEC)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
