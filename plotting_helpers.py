"""
Plotting helpers for use in the notebooks.

Two kinds of plots live here:
  - network/MNIST diagnostics: input digits, spike rasters (input / hidden /
    output layer), weight heatmaps, training curves
  - single-neuron voltage traces, for the persistent-memory pulse test

Phase-plane portraits are a separate concern and live in
phase_plane_helpers.py instead.

Every function takes plain torch tensors or numpy arrays and does its own
`.detach().cpu().numpy()` -- you never need to convert anything before
calling these.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ═══════════════════════════════════════════════════════════════════════════
# MNIST input / spike rasters
# ═══════════════════════════════════════════════════════════════════════════

def plot_input_image(img, ax=None, title="input digit"):
    """img: flat length-784 tensor/array (pixel intensities, not spikes)."""
    img = _to_numpy(img).reshape(28, 28)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img, cmap="gray_r")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_raster(spike_record, ax=None, title="", color="black", markersize=3):
    """Scatter raster for ONE sample: spike_record is (T, N) of 0/1 spikes.
    Each dot is one spike -- x = timestep, y = neuron index."""
    spikes = _to_numpy(spike_record)
    T, N = spikes.shape
    t_idx, n_idx = np.nonzero(spikes > 0.5)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(t_idx, n_idx, s=markersize, color=color, marker="|")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("neuron index")
    ax.set_title(title, fontsize=10)
    return ax


def get_sample_activity(net, loader, T, poisson_encode_fn, device, sample_idx=0):
    """Run one batch through `net` (whatever mode it's currently in -- call
    net.eval() first if you don't want dropout-like randomness, though this
    model has none) and pull out a single sample's full activity trace.

    Returns a dict: image, input, hidden, output (spike records for that one
    sample), and label.
    """
    imgs, targets = next(iter(loader))
    imgs_flat = imgs.view(imgs.shape[0], -1).to(device)
    spike_train = poisson_encode_fn(imgs_flat, T).to(device)

    with torch.no_grad():
        out = net(spike_train)

    return {
        'image':  imgs_flat[sample_idx],
        'input':  spike_train[:, sample_idx, :],
        'hidden': out['hidden'][:, sample_idx, :],
        'output': out['output'][:, sample_idx, :],
        'label':  targets[sample_idx].item(),
    }


def plot_network_rasters(activity, title_prefix=""):
    """Three-panel figure: input / hidden / output rasters for one sample,
    as returned by get_sample_activity()."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    plot_raster(activity['input'],  ax=axes[0], color="#444444",
                title=f"{title_prefix}input spikes  (784 pixels)")
    plot_raster(activity['hidden'], ax=axes[1], color="#1f77b4",
                title=f"{title_prefix}hidden layer spikes")
    plot_raster(activity['output'], ax=axes[2], color="#d62728",
                title=f"{title_prefix}output layer spikes  (true label = {activity['label']})")
    plt.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# weight heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def plot_weight_heatmap(W, ax=None, title=""):
    W = _to_numpy(W)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    vmax = np.abs(W).max() or 1.0
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_weight_heatmaps(net, title_prefix=""):
    """Side-by-side W1 (input->hidden) and W2 (hidden->output) heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_weight_heatmap(net.W1.weight, ax=axes[0], title=f"{title_prefix}W1  (input -> hidden)")
    plot_weight_heatmap(net.W2.weight, ax=axes[1], title=f"{title_prefix}W2  (hidden -> output)")
    plt.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# persistent-memory voltage traces (single neuron, pulse test)
# ═══════════════════════════════════════════════════════════════════════════

def plot_voltage_traces(step_idx, I_ext, v_traces, w_traces, spike_flags, dt_eff, title=""):
    """Three-row figure: I_ext pulse train, v (fast), w (slow), with spikes
    marked on the v trace. Same layout as the persistent-memory pulse test
    in notebook 1."""
    v_traces = _to_numpy(v_traces)
    w_traces = _to_numpy(w_traces)
    spike_flags = _to_numpy(spike_flags)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 2, 2]})
    ax_I, ax_v, ax_w = axes

    ax_I.plot(step_idx, I_ext, color="gray", lw=1.5)
    ax_I.fill_between(step_idx, I_ext, alpha=0.25, color="gray")
    ax_I.set_ylabel(r"$I_{ext}$", fontsize=11)

    ax_v.plot(step_idx, v_traces, color="steelblue", lw=1.5)
    spike_steps = np.where(spike_flags > 0.5)[0]
    ax_v.scatter(spike_steps, v_traces[spike_steps], color="black", s=15, zorder=3)
    ax_v.set_ylabel(r"$v$  (fast)", fontsize=11)

    ax_w.plot(step_idx, w_traces, color="darkorange", lw=1.5)
    ax_w.set_ylabel(r"$w$  (slow)", fontsize=11)
    ax_w.set_xlabel(f"network steps  (1 step = dt_eff = {dt_eff:.3g} model-time units)", fontsize=11)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.show()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# training curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, title="Training progress"):
    """Two-panel figure: loss (train vs test) and accuracy (train vs test),
    both against gradient step. `history` is the dict returned by
    training_helpers.train_fixed_steps()."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(history['step'], history['train_loss'], label="train", lw=1.5)
    ax_loss.plot(history['step'], history['test_loss'], label="test", lw=1.5)
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("loss")
    ax_loss.legend()

    ax_acc.plot(history['step'], np.array(history['train_acc']) * 100, label="train", lw=1.5)
    ax_acc.plot(history['step'], np.array(history['test_acc']) * 100, label="test", lw=1.5)
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.set_title("accuracy")
    ax_acc.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig, (ax_loss, ax_acc)
