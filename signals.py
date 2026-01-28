import numpy as np
from scipy.signal import chirp


def generate_chirp(duration, fs, f0, f1, method='linear'):
    n = int(duration * fs)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=duration, method=method)
    return t.astype(np.float32), signal.astype(np.float32)
    