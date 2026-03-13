# snap_server.py
from snap_spec.snap import UGRadioSnap

# Initialize the SNAP board server
snap = UGRadioSnap(host='0.0.0.0', is_discover=True)
snap.initialize(mode='corr', sample_rate=500)

print("SNAP server running. Waiting for client connections...")
snap.run()  # keeps the server alive to accept connections from clients like snap_test.py
