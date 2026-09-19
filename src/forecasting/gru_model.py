"""GRU-based multi-step workload forecaster."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1, horizon: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback) -> (batch, lookback, 1)
        x = x.unsqueeze(-1)
        out, h = self.gru(x)
        last = out[:, -1, :]  # (batch, hidden)
        return self.head(last)  # (batch, horizon)
