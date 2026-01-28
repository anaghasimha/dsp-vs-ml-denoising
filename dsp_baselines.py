import numpy as np

def moving_average_filter(x: np.ndarray, M: int) -> np.ndarray:
    if M <= 1:
        return x.astype(np.float32)

    h = np.ones(M, dtype=np.float32) / M
    y = np.convolve(x, h, mode="same")
    return y.astype(np.float32)


def wiener_filter(noisy_signal: np.ndarray, clean_signal: np.ndarray, fs: float, snr_db: float, eps: float = 1e-12):
    y = np.asarray(noisy_signal, dtype=np.float64)
    x = np.asarray(clean_signal, dtype=np.float64)
    N = len(y)
    if len(x) != N:
        raise ValueError("clean_signal and noisy_signal must have the same length")

    X = np.fft.rfft(x)
    Sx = (np.abs(X) ** 2) / (N * fs)  
    Px = np.mean(x ** 2)  
    sigma_n2 = Px / (10 ** (snr_db / 10))  
    Sn = sigma_n2 * np.ones_like(Sx)
    H = Sx / (Sx + Sn + eps)
    Y = np.fft.rfft(y)
    X_hat = H * Y
    denoised = np.fft.irfft(X_hat, n=N)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    return denoised.astype(np.float32), H.astype(np.float32), freqs