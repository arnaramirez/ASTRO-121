#!/usr/bin/env python
# coding: utf-8

# In[1]:


#didnt work, had the issue with the N>2 blocks
import ugradio
from ugradio import sdr
import numpy as np
import time


sample_rate = int(3.0e6)         
nblocks = 16                     
samples_per_block = 16384        
filename = f"noise_{sample_rate}Hz_{nblocks}blocks"


sdr = ugradio.sdr.SDR(
   direct=True,
   sample_rate=sample_rate
)


timestamps, data = sdr.capture_data(
   nblocks=nblocks,
   block_size=samples_per_block
)

print(f"Captured data shape: {data.shape}")


np.savez(
   filename,
   data=data,
   timestamps=timestamps,
   sample_rate=sample_rate,
   nblocks=nblocks,
   samples_per_block=samples_per_block,
   timestamp=time.time()
)

print(f"Noise data saved to {filename}.npz")

sdr.close()
del sdr


# In[ ]:




