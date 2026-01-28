import matplotlib.pyplot as plt

from signals import generate_chirp
from noise import add_white_noise

# Generating chirp
t, signal = generate_chirp(
    duration=10,
    fs=100,
    f0=1,
    f1=10
)

# Adding white noise
noisy_signal, noise = add_white_noise(signal, snr_db=-5)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label="Clean signal")
plt.plot(t, noisy_signal, label="Noisy signal", alpha=0.6)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Chirp + White Noise (10 dB SNR)")
plt.legend()
plt.grid(True)
plt.show()
