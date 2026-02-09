#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def compute_fft(x, fs, window='hann'):
    """
    Compute FFT of a time-domain signal x sampled at fs Hz.
    
    Parameters:
        x : array-like
            Time-domain signal.
        fs : float
            Sampling frequency in Hz.
        window : str or None
            Type of window function ('hann', 'hamming', or None).
            
    Returns:
        f : np.ndarray
            Frequency array, from -fs/2 to fs/2.
        X : np.ndarray
            FFT values (complex), shifted.
    """
    N = len(x)
    
    # Apply window
    if window == 'hann':
        w = np.hanning(N)
    elif window == 'hamming':
        w = np.hamming(N)
    else:
        w = np.ones(N)
    x_win = x * w
    
    # FFT
    X = np.fft.fftshift(np.fft.fft(x_win))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    
    return f, X


def voltage_spectrum(x, fs, window='hann'):
    """
    Return the voltage spectrum (magnitude) of x.
    """
    f, X = compute_fft(x, fs, window)
    V = np.abs(X) / len(x)  # normalize by N
    return f, V


def power_spectrum(x, fs, window='hann'):
    """
    Return the power spectrum of x.
    """
    f, V = voltage_spectrum(x, fs, window)
    P = V**2
    return f, P


def plot_spectrum(f, Y, type='power', label=None, color='C0'):
    """
    Plot a spectrum (power or voltage) with nice formatting.
    
    Parameters:
        f : array-like
            Frequency array.
        Y : array-like
            Spectrum values.
        type : str
            'power' or 'voltage' (for axis labels).
        label : str
            Legend label.
        color : str
            Plot color.
    """
    plt.figure(figsize=(8,4))
    plt.plot(f, Y, color=color, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power" if type=='power' else "Voltage")
    plt.title(f"{type.capitalize()} Spectrum")
    plt.grid(True)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()

