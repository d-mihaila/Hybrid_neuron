"""
Phase-plane portraits for the bistable neuron.

This is the 2-timescale (v, w) view -- i.e. the z=0 slice of the model, no
ultraslow third variable. Same fixed-point/stability conventions as the
bifurcation analysis: nullcline intersections, Jacobian trace/det for
stability, trajectories classified as -> rest or -> spiking.

For the 3-timescale (v, w, z) extension, see the TODO stub at the bottom of
this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from params import a, eps, c, d

# ── colour palette ───────────────────────────────────────────────────────────
C_VNULL  = '#111111'   # black  - v-nullcline
C_WNULL  = '#d45c1e'   # orange - w-nullcline
C_REST   = '#228833'   # green  - trajectories -> rest
C_SPIKE  = '#aa3377'   # purple - trajectories -> spiking / limit cycle
C_STABLE = '#111111'   # filled circle - stable fixed point
C_UNSTAB = '#ffffff'   # open circle   - unstable fixed point
C_SADDLE = '#e05050'   # cross         - saddle

lim = 5   # v, w axis half-range


# ═══════════════════════════════════════════════════════════════════════════
# Core helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_I_TC(a, w0):
    """Analytical transcritical current -- the I_eff at which the two fixed
    points merge (discriminant = 0). Closed form: I_TC = w0^2 / (1 - a^2)."""
    v_star = a * w0 / (1 - a**2)
    w_star = a * v_star + w0
    return w_star**2 - v_star**2


def find_fixed_points(a, w0, I_eff):
    """
    Intersections of the two nullclines at a given effective current I_eff.

    v-nullcline (dv/dt = 0):  w = +/- sqrt(v^2 + I_eff)
    w-nullcline (dw/dt = 0):  w = a*v + w0
    """
    A = a**2 - 1
    B = 2 * a * w0
    C = w0**2 - I_eff

    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return []
        v = -C / B
        return [(v, a * v + w0)]

    disc = B**2 - 4 * A * C
    disc_scale = B**2 + abs(4 * A * C)
    if disc_scale > 0 and abs(disc) / disc_scale < 1e-10:
        disc = 0.0

    if disc < 0:
        return []
    if disc == 0.0:
        v0 = -B / (2 * A)
        return [(v0, a * v0 + w0)]
    sq = np.sqrt(disc)
    v1 = (-B + sq) / (2 * A)
    v2 = (-B - sq) / (2 * A)
    return [(v1, a * v1 + w0), (v2, a * v2 + w0)]


def classify_fp(v, w, a, eps):
    """Stability from the 2x2 Jacobian: trace = 2v - eps, det = 2*eps*(a*w - v)."""
    trace = 2 * v - eps
    det = 2 * eps * (a * w - v)
    if det < 0:
        return 'saddle'
    elif trace < 0:
        return 'stable'
    else:
        return 'unstable'


def rhs_2d(t, state, a, w0, eps, I_eff):
    """2-D subsystem, z fixed at 0 (no ultraslow adaptation)."""
    v, w = state
    dv = v**2 - w**2 + I_eff
    dw = eps * (a * v - w + w0)
    return [dv, dw]


# ═══════════════════════════════════════════════════════════════════════════
# Single panel
# ═══════════════════════════════════════════════════════════════════════════

def _plot_single_panel(ax, w0, I_ext, v_lim=(-lim, lim)):
    """One phase-plane panel at a single I_ext, z=0."""
    w_lim = v_lim
    I_TC = compute_I_TC(a, w0)
    I_eff = I_ext + I_TC

    ax.set_facecolor('#ffffff')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # ── vector field ─────────────────────────────────────────────────────
    nq = 22
    vq = np.linspace(*v_lim, nq)
    wq = np.linspace(*w_lim, nq)
    Vq, Wq = np.meshgrid(vq, wq)
    DV = Vq**2 - Wq**2 + I_eff
    DW = eps * (a * Vq - Wq + w0)
    speed = np.sqrt(DV**2 + DW**2)
    speed[speed == 0] = 1e-12
    ax.quiver(Vq, Wq, DV / speed, DW / speed, speed, cmap='Greys', alpha=0.28,
              scale=28, width=0.003, pivot='mid')

    # ── nullclines ───────────────────────────────────────────────────────
    v_arr = np.linspace(*v_lim, 1200)
    inside = v_arr**2 + I_eff
    valid = inside >= 0
    w_pos = np.where(valid, np.sqrt(np.abs(inside)), np.nan)
    w_neg = np.where(valid, -np.sqrt(np.abs(inside)), np.nan)
    ax.plot(v_arr, w_pos, color=C_VNULL, lw=2.5, zorder=3)
    ax.plot(v_arr, w_neg, color=C_VNULL, lw=2.5, zorder=3)
    ax.plot(v_arr, a * v_arr + w0, color=C_WNULL, lw=2.5, ls='--', zorder=3)

    # ── fixed points, labelled stable / unstable / saddle ──────────────────
    fp_handles = []
    fps = find_fixed_points(a, w0, I_eff)
    for (vs, ws) in fps:
        if not (v_lim[0] <= vs <= v_lim[1] and w_lim[0] <= ws <= w_lim[1]):
            continue
        kind = classify_fp(vs, ws, a, eps)
        if kind == 'stable':
            ax.plot(vs, ws, 'o', color=C_STABLE, ms=11, mfc=C_STABLE, mew=2, zorder=6)
            fp_handles.append(mpatches.Patch(color=C_STABLE, label=f'stable  ({vs:.2f}, {ws:.2f})'))
        elif kind == 'unstable':
            ax.plot(vs, ws, 'o', color=C_STABLE, ms=11, mfc=C_UNSTAB, mew=2, zorder=6)
            fp_handles.append(mpatches.Patch(facecolor=C_UNSTAB, edgecolor=C_STABLE,
                                              label=f'unstable ({vs:.2f}, {ws:.2f})'))
        else:
            ax.plot(vs, ws, 'x', color=C_SADDLE, ms=13, mew=2.5, zorder=6)
            fp_handles.append(mpatches.Patch(color=C_SADDLE, label=f'saddle   ({vs:.2f}, {ws:.2f})'))

    # ── trajectories, classified spiking vs. rest ───────────────────────────
    T_END = 300.0
    V_SPIKE_THR = 1.5
    IC_BOUND = 3.0
    N_RANDOM = 6

    rng = np.random.default_rng(seed=42)
    random_ics = list(zip(rng.uniform(-IC_BOUND, IC_BOUND, N_RANDOM),
                           rng.uniform(-IC_BOUND, IC_BOUND, N_RANDOM)))
    special_ics = [(0.0, 0.0), (c, d)]
    all_ics = random_ics + special_ics
    reset_idx = len(all_ics) - 1   # (c, d) is always last

    def spikes(v_arr):
        half = len(v_arr) // 2
        return np.max(v_arr[half:]) > V_SPIKE_THR

    first_rest, first_spike = [True], [True]
    for idx, ic in enumerate(all_ics):
        sol = solve_ivp(rhs_2d, [0, T_END], list(ic), args=(a, w0, eps, I_eff),
                         method='RK45', max_step=0.05, rtol=1e-8, atol=1e-10)
        v_sol, w_sol = sol.y

        spiking = spikes(v_sol)
        color = C_SPIKE if spiking else C_REST
        lbl = None
        if spiking and first_spike[0]:
            lbl = '-> spiking'; first_spike[0] = False
        elif (not spiking) and first_rest[0]:
            lbl = '-> rest'; first_rest[0] = False

        mask = ((v_sol >= v_lim[0]) & (v_sol <= v_lim[1]) &
                (w_sol >= w_lim[0]) & (w_sol <= w_lim[1]))
        ax.plot(v_sol[mask], w_sol[mask], color=color, lw=1.3, alpha=0.72, zorder=4, label=lbl)

        marker_color = '#e87c1e' if idx == reset_idx else color
        ax.plot(ic[0], ic[1], 's', color=marker_color, ms=6, zorder=7, mec='black', mew=0.6)

    ax.set_xlim(*v_lim)
    ax.set_ylim(*w_lim)
    ax.axhline(0, color='#d0d0d0', lw=0.8, zorder=0)
    ax.axvline(0, color='#d0d0d0', lw=0.8, zorder=0)

    return fp_handles, I_TC, I_eff


# ═══════════════════════════════════════════════════════════════════════════
# Public function: 1x3 comparison across I_ext = 0, +I_ext, -I_ext
# ═══════════════════════════════════════════════════════════════════════════

def plot_input_sweep(w0, I_ext, savefile=None):
    """
    Three side-by-side phase planes, same w0, single (z=0) nullcline each:
    I_ext = 0, +I_ext, -I_ext.

    Fixed points are labelled stable / unstable / saddle; trajectories are
    classified as -> rest or -> spiking.

    Parameters
    ----------
    w0 : float
        Adaptation baseline, held fixed across all three panels.
    I_ext : float
        External current magnitude. Panels shown are 0, +I_ext, -I_ext.
    savefile : str, optional
        If given, saves the figure to this path (dpi=300).
    """
    I_cases = [0.0, abs(I_ext), -abs(I_ext)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True,
                              constrained_layout=True)
    fig.patch.set_facecolor('#ffffff')

    all_fp_handles = []
    for ax, I_val in zip(axes, I_cases):
        fp_handles, I_TC, I_eff = _plot_single_panel(ax, w0, I_val)
        sign = '0' if I_val == 0 else (f'+{I_val:g}' if I_val > 0 else f'{I_val:g}')
        ax.set_title(rf'$I_{{ext}}={sign}$    ($I_{{eff}}={I_eff:.1f}$)', fontsize=12, pad=10)
        all_fp_handles.extend(fp_handles)

    axes[0].set_ylabel(r'$w$  (recovery)', fontsize=12)
    for ax in axes:
        ax.set_xlabel(r'$v$  (voltage)', fontsize=12)

    # de-duplicate fixed-point legend entries (same label -> same fixed point)
    seen, dedup_fp = set(), []
    for h in all_fp_handles:
        if h.get_label() not in seen:
            seen.add(h.get_label())
            dedup_fp.append(h)

    legend_handles = [
        Line2D([0], [0], color=C_VNULL, lw=2.2, label=r'$\dot{v}=0$  (v-nullcline)'),
        mpatches.Patch(color=C_WNULL, label=r'$\dot{w}=0$  (w-nullcline)'),
        mpatches.Patch(color=C_REST, label='trajectory -> rest'),
        mpatches.Patch(color=C_SPIKE, label='trajectory -> spiking'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e87c1e',
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='reset IC  $(c,\\ d)$'),
    ] + dedup_fp

    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(rf'Phase planes for $w_0={w0}$, with I_TC = {I_TC:.1f}', fontsize=13, y=1.05)

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# TODO -- third (ultraslow) timescale, z
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase_plane_with_z(*args, **kwargs):
    """
    NOT YET IMPLEMENTED in this simplified handoff package.

    Everything above is the 2-timescale (fast v, slow w) model, i.e. the
    z = 0 slice. A 3-timescale extension -- adding a third, even-slower
    variable z on top of (v, w), e.g. z modulating w0 (or the reset (c, d))
    on a timescale eps_z << eps -- exists elsewhere but wasn't part of this
    handoff's source files, so it isn't ported into this package yet.

    To fill this in:
      1. Add a `rhs_3d(t, state, ...)` analogous to `rhs_2d` above, with a
         third dw/dt-style equation for z (and z entering wherever it
         modulates the (v, w) dynamics -- most likely inside w0 or I_TC).
      2. Extend `find_fixed_points` / `classify_fp` to 3x3 (the Jacobian
         stability argument no longer reduces to a simple trace/det check
         once z couples back into v or w).
      3. Build a plotting routine analogous to `_plot_single_panel` /
         `plot_input_sweep`, sliced at a few representative z values (or
         animated over z) rather than a single 2-D vector field.

    Ask Daria for the original script with the "ultraslow-z survey
    machinery" if you need the exact equations rather than reconstructing
    them from scratch.
    """
    raise NotImplementedError(
        "3-timescale (v, w, z) phase portraits aren't ported into this "
        "package yet -- see this function's docstring for how to add them."
    )
