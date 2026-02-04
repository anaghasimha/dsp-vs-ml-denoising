import numpy as np

from signals import generate_chirp
from noise import add_white_noise
from dsp_baselines import wiener_filter, moving_average_filter
from metrics import calculate_mse, calculate_snr

from ml_models import CNNDenoiser, WindowLinearModel

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



def build_dataset(duration, fs, snr_db, num_samples, seed=0):
    
    rng = np.random.default_rng(seed)
    data = []

    for _ in range(num_samples):
        # randomize chirp slightly
        f0 = float(rng.uniform(0.5, 2.0))
        f1 = float(rng.uniform(6.0, 12.0))

        _, x = generate_chirp(duration, fs, f0, f1)
        y, _ = add_white_noise(x, snr_db=snr_db, seed=int(rng.integers(0, 10**9)))

        data.append({"x": x.astype(np.float32), "y": y.astype(np.float32), "snr_db": float(snr_db)})

    return data


class PairDataset(Dataset):
    
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]["x"]  # [N]
        y = self.data[idx]["y"]  # [N]
        y_t = torch.from_numpy(y).unsqueeze(0)  # [1, N]
        x_t = torch.from_numpy(x).unsqueeze(0)  # [1, N]
        return y_t, x_t


def make_window_matrix(noisy_signal, win):
    
    assert win % 2 == 1, "win must be odd"
    half = win // 2
    ypad = np.pad(noisy_signal, (half, half), mode="reflect").astype(np.float32)

    N = len(noisy_signal)
    X = np.zeros((N, win), dtype=np.float32)
    for i in range(N):
        X[i, :] = ypad[i:i+win]
    return X  # [N, win]



# DSP evaluation (per sample)

def eval_dsp_methods(sample, fs):
    x = sample["x"]
    y = sample["y"]
    snr_db = sample["snr_db"]

    out = {}

    # Wiener 
    den_w, _, _ = wiener_filter(y, x, fs, snr_db)
    out["dsp_wiener_mse"] = float(calculate_mse(x, den_w))
    out["dsp_wiener_snr_out"] = float(calculate_snr(x, x - den_w))

    # Moving average
    den_ma = moving_average_filter(y, M=11)
    out["dsp_ma_mse"] = float(calculate_mse(x, den_ma))
    out["dsp_ma_snr_out"] = float(calculate_snr(x, x - den_ma))

    return out



# ML: training + evaluation
class TrainCfg:
    def __init__(self, epochs=5, lr=1e-3, batch_size=32, device=None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(model, train_data, cfg: TrainCfg):
    device = cfg.device
    model = model.to(device)

    ds = PairDataset(train_data)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, cfg.epochs + 1):
        total = 0.0
        for y, x in loader:
            y = y.to(device)  
            x = x.to(device)  

            opt.zero_grad()
            pred = model(y)
            loss = loss_fn(pred, x)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"[CNN] Epoch {ep:02d} | Avg Loss: {total / len(loader):.6f}")

    return model


@torch.no_grad()
def eval_cnn(model, sample, device):
    model.eval()
    y = torch.from_numpy(sample["y"]).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,N]
    pred = model(y).cpu().squeeze(0).squeeze(0).numpy().astype(np.float32)          # [N]
    return pred


def train_window_linear(model, train_data, win, cfg: TrainCfg):
    
    device = cfg.device
    model = model.to(device)

    X_list, t_list = [], []
    for s in train_data:
        X = make_window_matrix(s["y"], win)     
        t = s["x"].astype(np.float32).reshape(-1, 1) 
        X_list.append(X)
        t_list.append(t)

    X_all = np.vstack(X_list)  
    t_all = np.vstack(t_list)  

    
    X_t = torch.from_numpy(X_all).to(device)
    t_t = torch.from_numpy(t_all).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    B = cfg.batch_size
    num_rows = X_t.shape[0]

    for ep in range(1, cfg.epochs + 1):
        
        idx = torch.randperm(num_rows, device=device)
        X_shuf = X_t[idx]
        t_shuf = t_t[idx]

        total = 0.0
        steps = 0

        for start in range(0, num_rows, B):
            xb = X_shuf[start:start+B]
            tb = t_shuf[start:start+B]

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, tb)
            loss.backward()
            opt.step()

            total += loss.item()
            steps += 1

        print(f"[WIN] Epoch {ep:02d} | Avg Loss: {total / steps:.6f}")

    return model


@torch.no_grad()
def eval_window_linear(model, sample, win, device):
    model.eval()
    X = make_window_matrix(sample["y"], win)  
    X_t = torch.from_numpy(X).to(device)
    pred = model(X_t).cpu().numpy().astype(np.float32).reshape(-1)  
    return pred



# Main sweep

def main():
    
    duration = 10.0
    fs = 100.0

    
    test_snr_list = [-5, 0, 5, 10, 15, 20]

    
    train_snr_list = [5]          
    train_sizes = [200, 2000]     
    test_size = 200

    
    cfg = TrainCfg(epochs=5, lr=1e-3, batch_size=32)
    win = 11

    results = []

    for train_snr in train_snr_list:
        for train_n in train_sizes:
            
            train_data = build_dataset(duration, fs, train_snr, num_samples=train_n, seed=0)

            
            cnn = CNNDenoiser(channels=16, k=5)
            cnn = train_cnn(cnn, train_data, cfg)

            wlin = WindowLinearModel(win=win)
            wlin = train_window_linear(wlin, train_data, win, cfg)

            
            for test_snr in test_snr_list:
                test_data = build_dataset(duration, fs, test_snr, num_samples=test_size, seed=123)

                
                dsp_rows = [eval_dsp_methods(s, fs) for s in test_data]

                
                cnn_mse, cnn_snr_out = [], []
                win_mse, win_snr_out = [], []

                for s in test_data:
                    # CNN
                    den_cnn = eval_cnn(cnn, s, device=cfg.device)
                    cnn_mse.append(calculate_mse(s["x"], den_cnn))
                    cnn_snr_out.append(calculate_snr(s["x"], s["x"] - den_cnn))

                    # Window linear
                    den_win = eval_window_linear(wlin, s, win=win, device=cfg.device)
                    win_mse.append(calculate_mse(s["x"], den_win))
                    win_snr_out.append(calculate_snr(s["x"], s["x"] - den_win))

                row = {
                    "duration": duration,
                    "fs": fs,
                    "train_snr_db": float(train_snr),
                    "train_size": int(train_n),
                    "test_snr_db": float(test_snr),

                    # ML means
                    "ml_cnn_mse": float(np.mean(cnn_mse)),
                    "ml_cnn_snr_imp": float(np.mean(cnn_snr_out) - test_snr),
                    "ml_win_mse": float(np.mean(win_mse)),
                    "ml_win_snr_imp": float(np.mean(win_snr_out) - test_snr),

                    # DSP means
                    "dsp_wiener_mse": float(np.mean([d["dsp_wiener_mse"] for d in dsp_rows])),
                    "dsp_wiener_snr_imp": float(np.mean([d["dsp_wiener_snr_out"] for d in dsp_rows]) - test_snr),

                    "dsp_ma_mse": float(np.mean([d["dsp_ma_mse"] for d in dsp_rows])),
                    "dsp_ma_snr_imp": float(np.mean([d["dsp_ma_snr_out"] for d in dsp_rows]) - test_snr),
                }

                results.append(row)
                print("Finished:", row)

   
    import csv
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print("Saved results.csv")


if __name__ == "__main__":
    main()

