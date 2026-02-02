import numpy as np
import matplotlib.pyplot as plt

from signals import generate_chirp
from noise import add_white_noise
from dsp_baselines import wiener_filter, moving_average_filter
from metrics import calculate_mse, calculate_snr


# Experiment setup

duration = 10
fs = 100
f0, f1 = 1, 10

snr_in_list = [-5, 0, 5, 10, 15, 20]

# Generate clean signal once
t, clean_signal = generate_chirp(duration, fs, f0, f1)

# Storage
mse_wiener = []
mse_ma = []

snr_imp_wiener = []
snr_imp_ma = []


# Sweep over input SNR

for snr_db in snr_in_list:

    # Add noise
    noisy_signal, _ = add_white_noise(clean_signal, snr_db, seed=0)

    # Denoise
    den_wiener, _, _ = wiener_filter(
        noisy_signal, clean_signal, fs, snr_db
    )
    den_ma = moving_average_filter(noisy_signal, M=11)

    
    # MSE
    
    mse_w = calculate_mse(clean_signal, den_wiener)
    mse_m = calculate_mse(clean_signal, den_ma)

    mse_wiener.append(mse_w)
    mse_ma.append(mse_m)

    
    # Output SNR
    
    snr_out_w = calculate_snr(clean_signal, clean_signal - den_wiener)
    snr_out_m = calculate_snr(clean_signal, clean_signal - den_ma)

    
    # SNR improvement
    
    snr_imp_wiener.append(snr_out_w - snr_db)
    snr_imp_ma.append(snr_out_m - snr_db)


# Plot 1: SNR_in vs MSE

plt.figure(figsize=(7, 5))
plt.plot(snr_in_list, mse_wiener, marker="o", label="Wiener")
plt.plot(snr_in_list, mse_ma, marker="s", label="Moving Average")
plt.xlabel("Input SNR (dB)")
plt.ylabel("MSE")
plt.title("Input SNR vs MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: SNR_in vs SNR Improvement

plt.figure(figsize=(7, 5))
plt.plot(snr_in_list, snr_imp_wiener, marker="o", label="Wiener")
plt.plot(snr_in_list, snr_imp_ma, marker="s", label="Moving Average")
plt.xlabel("Input SNR (dB)")
plt.ylabel("SNR Improvement (dB)")
plt.title("Input SNR vs SNR Improvement")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
