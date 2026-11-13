"""Task-adapted sequence backbones: predict a log-growth correction.

PatchTST: channel-independent patch embedding/attention (arXiv:2211.14730).
iTransformer: attention across variate tokens (arXiv:2310.06625).
TSMixer: alternating temporal/channel MLPs (arXiv:2303.06053).
DLinear: moving-average trend/seasonal decomposition (arXiv:2205.13504).
These are adaptations for prefix-to-final cutoff regression, not claims to
reproduce the papers' original multistep benchmarks.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ['mlp', 'gru', 'lstm', 'tcn', 'transformer', 'patchtst',
                 'tsmixer', 'itransformer', 'dlinear']


def prefix_sequence(sample, length=64):
    """Interpolation sees only the observed prefix, never the event suffix."""
    curve, times, scores = sample['curve'], sample['times'], sample['scores']
    grid = np.linspace(curve.start, sample['now'], length)
    available = grid >= times[0]
    score = sample['score']
    cumulative = np.where(available, np.interp(grid, times, scores) / score, 0.)
    dt_hours = np.diff(grid) / 3600000
    relative_rate = np.r_[0., np.diff(cumulative) / dt_hours * sample['features']['elapsed_h']]
    relative_rate[~available] = 0.
    relative_rate[1:][~(available[1:] & available[:-1])] = 0.
    hour = (grid / 3600000 + 8) % 24
    weekday = ((grid / 86400000 + 3 + 8 / 24) % 7)
    sequence = np.column_stack([
        cumulative, np.clip(relative_rate, -8, 8),
        np.linspace(0., sample['progress'], length),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        available.astype(float),
    ])
    return sequence.astype(np.float32)


class CausalBlock(nn.Module):
    def __init__(self, inputs, width, dilation):
        super().__init__()
        self.one = nn.Conv1d(inputs, width, 3, padding=2 * dilation, dilation=dilation)
        self.two = nn.Conv1d(width, width, 3, padding=2 * dilation, dilation=dilation)
        self.skip = nn.Conv1d(inputs, width, 1) if inputs != width else nn.Identity()
        self.dropout = nn.Dropout(.15)

    def forward(self, x):
        length = x.shape[-1]
        y = self.dropout(F.gelu(self.one(x)[..., :length]))
        y = self.dropout(F.gelu(self.two(y)[..., :length]))
        return y + self.skip(x)


class MixerBlock(nn.Module):
    def __init__(self, length, channels):
        super().__init__()
        self.norm_time = nn.LayerNorm(channels)
        self.time = nn.Sequential(nn.Linear(length, length), nn.GELU(), nn.Dropout(.15), nn.Linear(length, length))
        self.norm_channel = nn.LayerNorm(channels)
        self.channel = nn.Sequential(nn.Linear(channels, 32), nn.GELU(), nn.Dropout(.15), nn.Linear(32, channels))

    def forward(self, x):
        x = x + self.time(self.norm_time(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel(self.norm_channel(x))


class Forecaster(nn.Module):
    def __init__(self, architecture, tab_features, width=48, length=64, channels=8):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        self.tab = nn.Sequential(nn.Linear(tab_features, width), nn.GELU(), nn.LayerNorm(width),
                                 nn.Dropout(.15), nn.Linear(width, width), nn.GELU())
        if architecture in ['gru', 'lstm']:
            recurrent = nn.GRU if architecture == 'gru' else nn.LSTM
            self.encoder = recurrent(channels, width, num_layers=2, dropout=.15, batch_first=True)
        elif architecture == 'tcn':
            self.encoder = nn.Sequential(*[CausalBlock(channels if i == 0 else width, width, 2 ** i)
                                           for i in range(5)])
        elif architecture in ['transformer', 'patchtst', 'itransformer']:
            input_width = {'transformer': channels, 'patchtst': 8, 'itransformer': length}[architecture]
            tokens = {'transformer': length, 'patchtst': (length - 8) // 4 + 1, 'itransformer': channels}[architecture]
            self.embedding = nn.Linear(input_width, width)
            self.position = nn.Parameter(torch.randn(1, tokens, width) * .01)
            layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2,
                                                dropout=.15, batch_first=True, norm_first=True,
                                                activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            if architecture == 'patchtst':
                self.channel_summary = nn.Linear(channels * width, width)
        elif architecture == 'tsmixer':
            self.encoder = nn.Sequential(*[MixerBlock(length, channels) for _ in range(3)])
            self.summary = nn.Linear(channels, width)
        elif architecture == 'dlinear':
            self.trend = nn.Linear(length, width)
            self.seasonal = nn.Linear(length, width)
            self.summary = nn.Linear(channels * width, width)
        inputs = width if architecture == 'mlp' else 2 * width
        self.head = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Dropout(.15), nn.Linear(width, 1))
        # Start from the known average-velocity extrapolation; learn its correction.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequence, tab):
        static = self.tab(tab)
        kind = self.architecture
        if kind == 'mlp':
            combined = static
        else:
            if kind in ['gru', 'lstm']:
                encoded, _ = self.encoder(sequence)
                dynamic = encoded[:, -1]
            elif kind == 'tcn':
                dynamic = self.encoder(sequence.transpose(1, 2))[:, :, -1]
            elif kind == 'transformer':
                dynamic = self.encoder(self.embedding(sequence) + self.position).mean(dim=1)
            elif kind == 'patchtst':
                batch, _, channels = sequence.shape
                patches = sequence.transpose(1, 2).unfold(-1, 8, 4)
                tokens = self.embedding(patches.reshape(batch * channels, -1, 8)) + self.position
                encoded = self.encoder(tokens).mean(dim=1).reshape(batch, -1)
                dynamic = self.channel_summary(encoded)
            elif kind == 'itransformer':
                dynamic = self.encoder(self.embedding(sequence.transpose(1, 2)) + self.position).mean(dim=1)
            elif kind == 'tsmixer':
                dynamic = self.summary(self.encoder(sequence).mean(dim=1))
            else:
                x = sequence.transpose(1, 2)
                trend = F.avg_pool1d(F.pad(x, (4, 4), mode='replicate'), kernel_size=9, stride=1)
                decomposition = self.trend(trend) + self.seasonal(x - trend)
                dynamic = self.summary(decomposition.flatten(1))
            combined = torch.cat([static, dynamic], dim=1)
        return self.head(combined).squeeze(-1)
