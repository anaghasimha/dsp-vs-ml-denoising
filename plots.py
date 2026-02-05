import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

def heatmap(df_sub, value_col, title):
    """
    df_sub must include columns: train_snr_db, test_snr_db, and value_col
    """
    pivot = df_sub.pivot_table(
        index="train_snr_db",
        columns="test_snr_db",
        values=value_col,
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    plt.xlabel("Test SNR (dB)")
    plt.ylabel("Train SNR (dB)")
    plt.title(title)

    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


# ---- choose one train_size slice at a time ----
for train_size in sorted(df["train_size"].unique()):
    sub = df[df["train_size"] == train_size].copy()

    heatmap(sub, "ml_cnn_mse", f"CNN MSE (train_size={train_size})")
    heatmap(sub, "ml_cnn_snr_imp", f"CNN SNR Improvement (train_size={train_size})")

    heatmap(sub, "ml_win_mse", f"Window Linear MSE (train_size={train_size})")
    heatmap(sub, "ml_win_snr_imp", f"Window Linear SNR Improvement (train_size={train_size})")

    heatmap(sub, "dsp_wiener_mse", f"Wiener MSE (train_size={train_size})")
    heatmap(sub, "dsp_wiener_snr_imp", f"Wiener SNR Improvement (train_size={train_size})")
