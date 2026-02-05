#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ugradio
from ugradio import sdr
import numpy as np
import time

sample_rate = int(3.0e6)
nblocks = 16
filename = f"noise_{sample_rate}Hz_{nblocks}blocks"

sdr = ugradio.sdr.SDR(
    direct=True,
    sample_rate=sample_rate
)

time.sleep(0.2)

blocks = []
timestamps = []

for i in range(nblocks):
    ts, data = sdr.capture_data(nblocks=1)
    blocks.append(data[0])       # one block
    timestamps.append(ts[0])     # matching timestamp

data = np.array(blocks)
timestamps = np.array(timestamps)

print("Captured data shape:", data.shape)

np.savez(
    filename,
    data=data,
    timestamps=timestamps,
    sample_rate=sample_rate,
    nblocks=nblocks,
    timestamp=time.time()
)

print(f"Noise data saved to {filename}.npz")

sdr.close()
del sdr

