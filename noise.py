import numpy as np

def add_white_noise(signal,snr_db,seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db / 10))

    noise = np.random.normal(0.0, np.sqrt(noise_power),size=signal.shape)

    noisy_signal = signal+noise
    return noisy_signal.astype(np.float32), noise.astype(np.float32)
    