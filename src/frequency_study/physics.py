"""Exact reference labels for Eq. (2)-(4) of the supplied manuscript.

All power inputs are signed net disturbances on a 200-MVA base. A positive
load increment yields a *negative* net input and therefore a frequency drop.
The sample at index k is held on [t_k, t_{k+1}); u[-1] is not integrated.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridConfig:
    f0_hz: float = 50.0
    s_base_mva: float = 200.0
    duration_s: float = 10.0
    n_points: int = 81
    f_limit_hz: float = 49.5
    rocof_limit_hz_s: float = 1.0

    @property
    def times(self):
        return np.linspace(0.0, self.duration_s, self.n_points)

    @property
    def dt(self):
        return self.duration_s / (self.n_points - 1)


def net_disturbance_pu(delta_load_mw, cfg=GridConfig()):
    """Positive MW load increment -> negative signed pu net disturbance."""
    return -np.asarray(delta_load_mw, dtype=np.float64) / cfg.s_base_mva


def exact_zoh(u_pu, h_s, damping, cfg=GridConfig()):
    """Return delta-f (Hz) using x[k+1]=a*x[k]+(1-a)*u[k]/D.

    The leading dimensions of u, H and D broadcast together. D is the
    dimensionless power/frequency slope in the paper, not MW/Hz.
    """
    u = np.asarray(u_pu, dtype=np.float64)
    if u.ndim < 1 or u.shape[-1] != cfg.n_points:
        raise ValueError(f"input must end in {cfg.n_points} time points")
    h = np.asarray(h_s, dtype=np.float64)
    d = np.asarray(damping, dtype=np.float64)
    if np.any(h <= 0) or np.any(d <= 0):
        raise ValueError("H and D must both be positive")
    shape = np.broadcast_shapes(u.shape[:-1], h.shape, d.shape)
    u = np.broadcast_to(u, shape + (cfg.n_points,))
    h = np.broadcast_to(h, shape)
    d = np.broadcast_to(d, shape)
    a = np.exp(-d * cfg.dt / (2.0 * h))
    gain = -np.expm1(-d * cfg.dt / (2.0 * h)) / d
    x = np.zeros(shape + (cfg.n_points,), dtype=np.float64)
    for k in range(cfg.n_points - 1):
        x[..., k + 1] = a * x[..., k] + gain * u[..., k]
    return cfg.f0_hz * x


def hard_indicators(delta_f_hz, cfg=GridConfig()):
    """Sampled window minimum and maximum absolute forward-difference RoCoF."""
    y = np.asarray(delta_f_hz, dtype=np.float64)
    if y.shape[-1] != cfg.n_points:
        raise ValueError("frequency trace has wrong number of time points")
    f_min = np.min(cfg.f0_hz + y, axis=-1)
    r_max = np.max(np.abs(np.diff(y, axis=-1) / cfg.dt), axis=-1)
    unsafe = (f_min < cfg.f_limit_hz) | (r_max > cfg.rocof_limit_hz_s)
    return f_min, r_max, unsafe


def positive_load_step(delta_mw, h_s, damping, cfg=GridConfig()):
    """Analytical step response for an independent validation of exact_zoh."""
    p = net_disturbance_pu(delta_mw, cfg)
    return cfg.f0_hz * (p / damping) * (1 - np.exp(-cfg.times * damping / (2 * h_s)))
