import numpy as np
import time
from snap_spec.snap import UGRadioSnap

print("Connecting to SNAP board...")

spec = UGRadioSnap(host='localhost', is_discover=True)

print("Initializing correlator...")
spec.initialize(mode='corr', sample_rate=500)

print("Reading spectra...")

prev_acc = None

for i in range(10):

    data = spec.read_data(prev_acc)

    prev_acc = data['acc_cnt']

    print("Integration:", prev_acc)
    print("Cross spectrum shape:", np.shape(data['cross']))

    time.sleep(0.1)

print("Test complete.")
