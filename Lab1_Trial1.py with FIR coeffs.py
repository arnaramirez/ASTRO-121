#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ugradio
from ugradio import sdr
import numpy as np
import time

sample_rate = int(3.0e6)
filename = f"testing_alias_{sample_rate}Hz"

sdr = ugradio.sdr.SDR(direct=True, sample_rate=sample_rate)

# disabling anti-aliasing FIR filter
fir_coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 2047],
                      dtype=np.int16)
sdr.set_fir_coeffs(fir_coeffs)

time.sleep(0.2)  # let buffers settle

# capture data
data = sdr.capture_data(nblocks=2)[1]

np.savez(
    filename,
    data=data,
    sample_rate=sample_rate,
    fir_coeffs=fir_coeffs,
    timestamp=time.time()
)

print(f"Data saved to {filename}")

sdr.close()
del sdr


# In[ ]:




