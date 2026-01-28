import numpy as np
import matplotlib.pyplot as plt

from signals import generate_chirp
from noise import add_white_noise
from dsp_baselines import wiener_filter, moving_average_filter

duration = 10
fs = 100
f0, f1 = 1, 10
snr_db = 10

t, clean = generate_chirp(duration, fs, f0, f1)
noisy, noise = add_white_noise(clean, snr_db=snr_db, seed=0)

denoised_w, H, freqs = wiener_filter(noisy, clean, fs, snr_db)
denoised_ma = moving_average_filter(noisy, M=11)

mse_w = np.mean((denoised_w - clean) ** 2)
mse_ma = np.mean((denoised_ma - clean) ** 2)

print("MSE Wiener:", mse_w)
print("MSE MovAvg:", mse_ma)

plt.figure(figsize=(10, 4))
plt.plot(t, clean, label="Clean")
plt.plot(t, noisy, label="Noisy", alpha=0.5)
plt.plot(t, denoised_w, label="Wiener")
plt.plot(t, denoised_ma, label="Moving Avg", alpha=0.8)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(freqs, H)
plt.title("Wiener Transfer Function H(f)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Gain")
plt.grid(True)
plt.show()
