"""
The single bistable neuron -- the core model everything else builds on.

Two-timescale hybrid spiking neuron: a fast voltage-like variable `v` and a
slower recovery variable `w`, integrated with RK4, with a hard
threshold-and-reset when v crosses v_th.

    dv/dt = v^2 - w^2 + I + I_TC(w0)
    dw/dt = eps * (a*v - w + w0)

    if v >= v_th:  v -> c,  w -> d

`I_TC` is the transcritical current that keeps the neuron's excitable
operating point pinned near I_eff = 0 regardless of w0 -- see
`compute_I_TC` in phase_plane_helpers.py for where that closed form comes
from, and bifurcation-analysis notes for the full story.

This file has no torch.nn / dataset / training dependencies -- it only needs
torch tensors in and out, so it works standalone for the phase-plane and
persistent-memory notebook, and is imported by network_model.py for the full
MNIST network.
"""
import numpy as np
import torch
from params import a, eps, c, d, v_th, w0, dt as _dt


def bistable_neuron(v, w, I, dt=_dt, steps=5, w0=None):
    """Advance (v, w) by one network "timestep" (= steps * dt of ODE time),
    in a single RK4 step of size dt_eff = dt * steps.

    v, w, I : torch tensors, any matching shape (scalar, or (batch, N))
    w0      : scalar or tensor broadcastable against v/w. Leave as None to
              use this module's `w0` (set from params.py at import time --
              change it with e.g. `neuron_model.w0 = -0.5` to affect every
              call that doesn't pass w0 explicitly). Pass a per-neuron
              tensor here instead of the module default if you want each
              neuron to have its own w0.

    Returns (v, w, spiked) where `spiked` is 1.0 on the timesteps that
    crossed threshold this call, else 0.0 -- same shape as v.
    """
    if w0 is None:
        w0 = globals()['w0']

    I_TC = w0**2 / (1 - a**2)  # recomputed from whichever w0 was passed in

    def dv(v_, w_, I_):
        return v_**2 - w_**2 + I_ + I_TC

    def dw(v_, w_):
        return eps * (a * v_ - w_ + w0)

    dt_eff = dt * steps

    kv1 = dv(v,               w,               I)
    kw1 = dw(v,               w)

    kv2 = dv(v + dt_eff/2*kv1, w + dt_eff/2*kw1, I)
    kw2 = dw(v + dt_eff/2*kv1, w + dt_eff/2*kw1)

    kv3 = dv(v + dt_eff/2*kv2, w + dt_eff/2*kw2, I)
    kw3 = dw(v + dt_eff/2*kv2, w + dt_eff/2*kw2)

    kv4 = dv(v + dt_eff*kv3,   w + dt_eff*kw3,   I)
    kw4 = dw(v + dt_eff*kv3,   w + dt_eff*kw3)

    v = v + (dt_eff / 6) * (kv1 + 2*kv2 + 2*kv3 + kv4)
    w = w + (dt_eff / 6) * (kw1 + 2*kw2 + 2*kw3 + kw4)

    # ── threshold crossing & reset ──────────────────────────────────────────
    fired_soft = spike_fn(v, v_th)
    fired = (fired_soft > 0.5)

    v = torch.where(fired, torch.full_like(v, c), v)
    w = torch.where(fired, torch.full_like(w, d), w)

    return v, w, fired_soft   # spikes: 0. or 1.


class SpikeFunction(torch.autograd.Function):
    """
    Forward:  Heaviside  S = 1 if v >= v_th, else 0
    Backward: ATan surrogate  dS/dv = 1/pi * 1/(1 + (v*pi)^2)
    (a smooth stand-in for the true derivative of the Heaviside step, so
    gradients can flow through spikes during training -- same trick used in
    snnTorch and most surrogate-gradient SNN libraries.)
    """
    @staticmethod
    def forward(ctx, v, v_th):
        spk = (v >= v_th).float()
        ctx.save_for_backward(v - v_th)   # shift so threshold sits at 0
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (shifted_v,) = ctx.saved_tensors
        surrogate = 1 / (1 + (np.pi * shifted_v).pow(2)) / np.pi
        return grad_output * surrogate, None   # None: v_th isn't learnable


spike_fn = SpikeFunction.apply
