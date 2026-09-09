"""
All model + training parameters in one place.

Everything else in this package (neuron_model.py, network_model.py,
training_helpers.py, phase_plane_helpers.py, and both notebooks) imports its
constants from here. Change a value here and it propagates everywhere -- you
should not need to hunt through other files for hard-coded numbers.
"""

# ── bistable neuron parameters ───────────────────────────────────────────────
# dv/dt = v^2 - w^2 + I + I_TC(w0)
# dw/dt = eps * (a*v - w + w0)
# reset on spike (v >= v_th):  v -> c,  w -> d
v_th  = 1     # spike threshold on v
eps   = 0.5   # timescale of the recovery variable w (smaller = slower w)
a     = 0.1   # sensitivity of w to subthreshold v
c     = 0     # reset value of v after a spike
d     = 1     # reset value of w after a spike
w_0   = 1     # adaptation baseline -- the main "personality knob" of the neuron
w0    = w_0   # alias used throughout the code (kept for backward compatibility)

# ── integration ──────────────────────────────────────────────────────────────
dt    = 0.01  # RK4 step size (abstract time units, not ms -- see notebook 1)

# ── network architecture ─────────────────────────────────────────────────────
N_in     = 28 * 28   # MNIST pixels
N_hidden = 100
N_out    = 10

# Zenke-style init std, offered as an alternative to the default normal(0, 0.5)
# init in network_model.py -- see the commented-out option in Network.__init__.
std_hidden = (2 / N_hidden) ** 0.5
std_out    = (2 / N_out) ** 0.5

# ── network timing (in "network steps" -- one bistable_neuron() call each) ──
T          = 50    # stimulus-presentation length
T_wait     = 100   # silent steps appended after the stimulus (memory window)
T_eval     = 100   # how many of the final steps are summed into the vote

# ── training ──────────────────────────────────────────────────────────────
TOTAL_STEPS = 10000   # full-run gradient steps (see notebook 2 for a shorter demo default)
EVAL_EVERY  = 50      # evaluate on the full test set every N steps
batch_size  = 128
lr          = 5e-4
