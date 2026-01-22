import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

t = np.linspace(0,10,1000)
waveform = chirp(t,f0=1,f1=10,t1=10,method='linear')

plt.figure(figsize=(10,8))
plt.plot(t,waveform,color='blue')
plt.title('Chirp Signal')
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.grid(True)