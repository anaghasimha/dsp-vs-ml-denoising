
import torch
import torch.nn as nn
class CNNDenoiser(nn.Module):
    def __init__(self, channels=16, k=5):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=k, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=k, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class WindowLinearModel(nn.Module):
    
    def __init__(self, win: int):
        super().__init__()
        self.fc = nn.Linear(win, 1)  

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        
        return self.fc(feat)
