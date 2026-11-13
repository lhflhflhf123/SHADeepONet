"""PI-DeepONet (27,745 parameters) and two explicit sequence baselines."""

import torch
from torch import nn


VARIANTS = {
    "data_only": (False, False, "none", False, False),
    "pi_baseline": (False, False, "differential", False, False),
    "hard_ic": (True, False, "differential", False, False),
    "structure": (True, True, "differential", False, False),
    "hybrid": (True, True, "hybrid", False, False),
    "risk_heads": (True, True, "hybrid", True, False),
    "full_ema": (True, True, "hybrid", True, True),
}


def encoder(in_features):
    layers = []
    for a, b in zip([in_features, 48, 48, 48], [48, 48, 48, 48]):
        layers.extend([nn.Linear(a, b), nn.Tanh()])
    layers.pop()  # first THREE affine layers are tanh; the fourth is linear
    return nn.Sequential(*layers)


class PIDeepONet(nn.Module):
    def __init__(self, variant="full_ema", n_points=81, duration_s=10.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.hard_ic, self.structured, self.physics_mode, self.risk_heads, self.ema = VARIANTS[variant]
        self.duration_s = duration_s
        self.branch = encoder(n_points)
        self.parameter_encoder = encoder(2)
        self.trunk = encoder(1)
        self.fusion = nn.Sequential(nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def core(self, branch, eta, times):
        b = self.branch(branch)
        p = self.parameter_encoder(eta)
        t = self.trunk(2.0 * times[:, None] / self.duration_s - 1.0)
        return self.fusion(b[:, None, :] * p[:, None, :] * t[None, :, :]).squeeze(-1)

    def forward(self, u, eta, times):
        if self.structured:
            amplitude = u.square().mean(dim=-1, keepdim=True).sqrt()
            shape = u / amplitude.clamp_min(1e-20)
            values = (self.core(shape, eta, times) - self.core(-shape, eta, times)) / 2
            return amplitude * times[None, :] * values
        out = self.core(u / 0.20, eta, times)
        return out * times[None, :] if self.hard_ic else out


class LSTMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, 1)

    def forward(self, u, eta, times):
        n, m = u.shape
        t = (2 * times / times[-1] - 1).expand(n, m)
        x = torch.cat([u[..., None] / .20, eta[:, None].expand(n, m, 2), t[..., None]], dim=-1)
        return self.out(self.rnn(x)[0]).squeeze(-1)


class SpectralBlock(nn.Module):
    def __init__(self, width=32, modes=16):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * .01)
        self.linear = nn.Conv1d(width, width, 1)

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        out = torch.zeros_like(xf)
        modes = min(self.modes, xf.shape[-1])
        out[..., :modes] = torch.einsum("bim,iom->bom", xf[..., :modes], self.weight[..., :modes])
        return torch.nn.functional.gelu(torch.fft.irfft(out, n=x.shape[-1], dim=-1) + self.linear(x))


class FNOBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = nn.Conv1d(4, 32, 1)
        self.blocks = nn.Sequential(*(SpectralBlock(32, 16) for _ in range(4)))
        self.out = nn.Sequential(nn.Conv1d(32, 32, 1), nn.GELU(), nn.Conv1d(32, 1, 1))

    def forward(self, u, eta, times):
        n, m = u.shape
        t = (2 * times / times[-1] - 1).expand(n, m)
        x = torch.cat([u[..., None] / .20, eta[:, None].expand(n, m, 2), t[..., None]], dim=-1)
        return self.out(self.blocks(self.lift(x.transpose(1, 2)))).squeeze(1)


def make_model(variant):
    if variant == "lstm":
        return LSTMBaseline()
    if variant == "fno":
        return FNOBaseline()
    return PIDeepONet(variant)
