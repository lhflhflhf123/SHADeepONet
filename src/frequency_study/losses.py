"""Discrete differential/integral defects and bounded smooth extrema."""

import math

import torch


def hard_indicators_torch(y, dt=0.125, f0=50.0, f_lim=49.5, r_lim=1.0):
    f_min = (f0 + y).amin(dim=-1)
    r_max = ((y[:, 1:] - y[:, :-1]) / dt).abs().amax(dim=-1)
    return f_min, r_max, (f_min < f_lim) | (r_max > r_lim)


def smooth_indicators(y, dt=0.125, f0=50.0, beta_f=1000.0, beta_r=500.0):
    f_min = -torch.logsumexp(-beta_f * (f0 + y), dim=-1) / beta_f
    r = ((y[:, 1:] - y[:, :-1]) / dt).abs()
    r_max = torch.logsumexp(beta_r * r, dim=-1) / beta_r
    return f_min, r_max


def physics_defects(y, u, h, d, dt=0.125, f0=50.0, smooth_threshold=1e-5):
    ybar = (y[:, 1:] + y[:, :-1]) / 2
    ubar = (u[:, 1:] + u[:, :-1]) / 2
    s = (y[:, 1:] - y[:, :-1]) / dt - (f0 * u[:, :-1] - d[:, None] * ybar) / (2 * h[:, None])
    integral = y[:, 1:] - y[:, :-1] - dt * (f0 * ubar - d[:, None] * ybar) / (2 * h[:, None])
    smooth = (u[:, 1:] - u[:, :-1]).abs() < smooth_threshold
    ls = (s[smooth] / 1.0).square().sum() / (smooth.sum() + 1)
    li = (integral[~smooth] / .05).square().sum() / ((~smooth).sum() + 1)
    return ls, li


def component_losses(y, truth, u, h, d, model):
    parts = {"d": (y - truth).square().mean()}
    if model.physics_mode != "none":
        parts["s"], integral = physics_defects(y, u, h, d)
        if model.physics_mode == "hybrid":
            parts["I"] = integral
    if model.risk_heads:
        fn, rn = smooth_indicators(y)
        ft, rt = smooth_indicators(truth)
        parts["n"] = ((fn - ft) / .05).square().mean()
        parts["r"] = ((rn - rt) / .10).square().mean()
    return parts


class EMALossWeights:
    def __init__(self, decay=.9, epsilon=1e-8):
        self.decay, self.epsilon = decay, epsilon
        self.mean = {}

    def combine(self, parts):
        for name, value in parts.items():
            current = float(value.detach())
            self.mean[name] = current if name not in self.mean else self.decay * self.mean[name] + (1 - self.decay) * current
        inv = {name: 1 / (self.mean[name] + self.epsilon) for name in parts}
        denom = sum(inv.values())
        weights = {name: len(parts) * inv[name] / denom for name in parts}
        return sum(parts[name] * weights[name] for name in parts)


def extrema_gap_bounds(n_points=81, beta_f=1000.0, beta_r=500.0):
    return math.log(n_points) / beta_f, math.log(n_points - 1) / beta_r
