import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return out + res


class TCN_NoAttention(nn.Module):
    """
    TCN without temporal attention (for ablation study)
    """
    def __init__(self, input_dim=36, num_classes=2):
        super().__init__()

        channels = [64, 128, 256]
        layers = []

        for i, ch in enumerate(channels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_ch, ch,
                    kernel=3,
                    dilation=2 ** i,
                    dropout=0.3
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        feat = self.tcn(x)              # (B, C, T)
        pooled = self.pool(feat).squeeze(-1)
        return self.fc(pooled)
