#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load Channel 1 data
# -------------------------
d = np.load("IF_from_KeysightLO10MHz_RF9p8MHz.npz")
x  = d["data"]
fs = float(d["sample_rate"])

# -------------------------
# Time axis
# -------------------------
t = np.arange(len(x)) / fs

# -------------------------
# FFT (no custom packages)
# -------------------------
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), d=1/fs)
P = np.abs(X)**2

# Shift for centered spectrum
freqs = np.fft.fftshift(freqs)
P = np.fft.fftshift(P)

# -------------------------
# Measured IF frequency
# -------------------------
f_measured = abs(freqs[np.argmax(P)])

print("Sampling rate:", fs, "Hz")
print("Measured IF frequency:", f_measured, "Hz")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,4))

# Time-domain (zoomed)
plt.subplot(1,2,1)
plt.plot(t[:300], x[:300])
plt.xlabel("Time (s)")
plt.xlim(0, 0.00005)
plt.ylabel("Voltage (arb)")
plt.title("Channel 1 IF (time domain)")

# Frequency-domain
plt.subplot(1,2,2)
plt.plot(freqs, P)
plt.xlim(-500_000, 500_000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power |FFT|² (arb)")
plt.ylim(0, 1e3)
plt.title("Channel 1 IF Spectrum")

plt.tight_layout()
plt.show()


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load Channel 1 data
# -------------------------
d = np.load("IF_from_KeysightN9310A_10MHz_delayedLO_RF9p8MHz_CH2.npz")
x  = d["data"]
fs = float(d["sample_rate"])

# -------------------------
# Time axis
# -------------------------
t = np.arange(len(x)) / fs

# -------------------------
# FFT (no custom packages)
# -------------------------
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), d=1/fs)
P = np.abs(X)**2

# Shift for centered spectrum
freqs = np.fft.fftshift(freqs)
P = np.fft.fftshift(P)

# -------------------------
# Measured IF frequency
# -------------------------
f_measured = abs(freqs[np.argmax(P)])

print("Sampling rate:", fs, "Hz")
print("Measured IF frequency:", f_measured, "Hz")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,4))

# Time-domain (zoomed)
plt.subplot(1,2,1)
plt.plot(t[:300], x[:300])
plt.xlabel("Time (s)")
plt.ylabel("Voltage (arb)")
plt.title("Channel 1 IF (time domain)")

# Frequency-domain
plt.subplot(1,2,2)
plt.plot(freqs, P)
plt.xlim(-500_000, 500_000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power |FFT|² (arb)")
plt.title("Channel 1 IF Spectrum")

plt.tight_layout()
plt.show()


# In[ ]:




