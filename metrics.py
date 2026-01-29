import numpy as np


def calculate_mse(clean_signal, denoised_signal):
    return np.mean((clean_signal-denoised_signal)**2)

def calculate_snr(signal,error):
    return 10 * np.log10(
        np.mean(signal ** 2) / np.mean(error ** 2)
    )

