import pyedflib, numpy as np

f = pyedflib.EdfReader("../mesa-commercial-use/polysomnography/edfs/mesa-sleep-0001.edf")

idx = [i for i,l in enumerate(f.getSignalLabels()) if "PLETH" in l.upper()][0]

sig = f.readSignal(idx)

print(sig.min(), sig.max())