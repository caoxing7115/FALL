import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        score = self.attn(x)    # (B, T, 1)
        alpha = torch.softmax(score, dim=1)
        out = (x * alpha).sum(dim=1)
        return out, alpha


class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
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


class TCN_Attention(nn.Module):
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
        self.attn = TemporalAttention(channels[-1])
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        feat = self.tcn(x)
        pooled, alpha = self.attn(feat)
        out = self.fc(pooled)
        return out, alpha
