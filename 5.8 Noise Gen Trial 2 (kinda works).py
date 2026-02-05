#!/usr/bin/env python
# coding: utf-8

# In[1]:


#runs, stacking for N>2 but gives wonky histos

import ugradio
from ugradio import sdr
import numpy as np
import time

sample_rate = int(3.0e6)
samples_per_block = 16384
n_captures = 16    

sdr = ugradio.sdr.SDR(direct=True, sample_rate=sample_rate)
time.sleep(0.2)

blocks = []

for i in range(n_captures):
    _, data = sdr.capture_data(
        nblocks=1,
        block_size=samples_per_block
    )
    blocks.append(data[0])

blocks = np.array(blocks)

sdr.close()
del sdr

np.savez(
    f"noise_{sample_rate}Hz_{n_captures}blocks",
    data=blocks,
    sample_rate=sample_rate
)

print("Captured shape:", blocks.shape)


# In[ ]:




