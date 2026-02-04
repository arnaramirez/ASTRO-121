#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import os

fss = [1000000.0, 2000000.0, 2500000.0, 3000000.0, 3200000.0]               
Ns = [2048, 4096, 8192, 16384]         

data_dir = "."
pad_factor = 16
span_hz = 5000

def load_npz(fs, N):
    fn = f"FreqRes1000000.0Hz, {N}.npz"      
    path = os.path.join(data_dir, fn)
    d = np.load(path)
    x = d["data"].astype(float)
    fs_file = float(d["sample_rate"]) if "sample_rate" in d.files else fs
    return x, fs_file, path

def centered_spectrum(x, fs, pad_factor=1):
    x = x - np.mean(x)
    N = len(x)
    Nz = int(pad_factor * N)
    X = np.fft.fft(x, n=Nz)
    f = np.fft.fftfreq(Nz, d=1/fs)

    Xs = np.fft.fftshift(X)
    fs_ = np.fft.fftshift(f)
    P = (np.abs(Xs)**2) / (N**2)     # normalized power (counts^2)
    return fs_, Xs, P

for N in Ns:
    
    for fs in fss:
        x, fs_file, path = load_npz(fs, N)

        f, Xs, P = centered_spectrum(x, fs_file, pad_factor=pad_factor)

        idx_peak = np.argmax(P)
        f_center = f[idx_peak]

        m = (f >= f_center - span_hz) & (f <= f_center + span_hz)

        plt.figure(figsize=(9,3))
        plt.plot(f[m] - f_center, P[m])
        plt.xlabel("Frequency offset from strongest peak (Hz)")
        plt.ylabel("Power (counts², normalized)")
        plt.title(f"Two-tone spectrum (linear) — N={N}, fs={fs/1e6:.2f} MHz")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Log power plot (dB)
        plt.figure(figsize=(9,3))
        plt.plot(f[m] - f_center, 10*np.log10(P[m] + 1e-12))
        plt.xlabel("Frequency offset from strongest peak (Hz)")
        plt.ylabel("Power (dB, relative)")
        plt.title(f"Two-tone spectrum (dB) — N={N}, fs={fs/1e6:.2f} MHz")
        plt.grid(True, which="both")
        plt.tight_layout()
    plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import os

fss = [1000000.0, 2000000.0, 2500000.0, 3000000.0, 3200000.0]               
Ns = [2048, 4096, 8192, 16384]         

data_dir = "."
pad_factor = 16
span_hz = 5000

def load_npz(fs, N):
    fn = f"FreqRes1000000.0Hz, {N}.npz"      
    path = os.path.join(data_dir, fn)
    d = np.load(path)
    x = d["data"].astype(float)
    fs_file = float(d["sample_rate"]) if "sample_rate" in d.files else fs
    return x, fs_file, path

def centered_spectrum(x, fs, pad_factor=1):
    x = x - np.mean(x)
    N = len(x)
    Nz = int(pad_factor * N)
    X = np.fft.fft(x, n=Nz)
    f = np.fft.fftfreq(Nz, d=1/fs)

    Xs = np.fft.fftshift(X)
    fs_ = np.fft.fftshift(f)
    P = (np.abs(Xs)**2) / (N**2)     # normalized power (counts^2)
    return fs_, Xs, P

for N in Ns:
    # Create subplots: 2 rows (linear + dB), len(fss) columns
    fig, axs = plt.subplots(2, len(fss), figsize=(4*len(fss), 6), sharey='row')
    
    for i, fs in enumerate(fss):
        x, fs_file, path = load_npz(fs, N)
        f, Xs, P = centered_spectrum(x, fs_file, pad_factor=pad_factor)

        idx_peak = np.argmax(P)
        f_center = f[idx_peak]
        m = (f >= f_center - span_hz) & (f <= f_center + span_hz)
        f_offset = f[m] - f_center

        # Linear power
        axs[0, i].plot(f_offset, P[m])
        axs[0, i].set_title(f"fs={fs/1e6:.2f} MHz")
        axs[0, i].set_xlabel("Frequency offset (Hz)")
        axs[0, i].grid(True)

        # dB power
        axs[1, i].plot(f_offset, 10*np.log10(P[m] + 1e-12))
        axs[1, i].set_xlabel("Frequency offset (Hz)")
        axs[1, i].grid(True, which='both')

        if i == 0:
            axs[0, i].set_ylabel("Power (counts²)")
            axs[1, i].set_ylabel("Power (dB)")

    fig.suptitle(f"Two-tone spectra — N={N}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

