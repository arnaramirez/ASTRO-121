#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ugradio
from ugradio import sdr
import numpy as np
import time

sample_rate = int(1.0e6)
samples_per_block = 16384
n_captures = 2

sdr = ugradio.sdr.SDR(direct=True, sample_rate=sample_rate,)
time.sleep(0.2)

data = sdr.capture_data(nblocks=n_captures, nsamples = samples_per_block)


sdr.close()
del sdr

np.savez(
    f"noise_{sample_rate}Hz_{n_captures}blocks, {samples_per_block}samples",
    data=data[1:],
    sample_rate=sample_rate
)

print("Captured shape:", data.shape)
