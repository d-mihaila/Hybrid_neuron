"""
Two-layer feedforward spiking network built from bistable_neuron.

    input (784 Poisson-encoded pixels) -> W1 -> hidden layer -> W2 -> output layer

Both the hidden and output layers are populations of the same bistable
neuron defined in neuron_model.py. W1 and W2 are the only trainable
parameters (plain linear layers, no bias); the neuron dynamics themselves
are not learned.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from params import N_in, N_hidden, N_out, T_wait, T_eval, batch_size
from neuron_model import bistable_neuron

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42  # fixed seed so weight initialisation is reproducible run to run

def set_seed(seed=SEED):
    """Reset all RNGs before building a Network(), so its initial weights are
    reproducible. Call this right before `Network()` if you want two runs
    (e.g. before/after changing a hyperparameter) to start from the same
    initialisation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(N_in, N_hidden, bias=None)
        self.W2 = nn.Linear(N_hidden, N_out, bias=None)

        # default initialisation
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.W2.weight, mean=0.0, std=0.5)

        # -- alternative initialisations, uncomment to try --
        # nn.init.uniform_(self.W1.weight, a=-0.5, b=0.5)
        # nn.init.uniform_(self.W2.weight, a=-0.5, b=0.5)
        #
        # from params import std_hidden, std_out       # Zenke-style init
        # nn.init.normal_(self.W1.weight, mean=0.0, std=std_hidden)
        # nn.init.normal_(self.W2.weight, mean=0.0, std=std_out)

        # per-timestep neuron state, allocated in reset_state()
        self.v1 = self.w1 = None
        self.v2 = self.w2 = None

        import neuron_model
        print('this network operates with w0 =', neuron_model.w0)

        self.to(device=device)

    def reset_state(self, batch_size):
        """Call before processing each new batch of images -- resets every
        neuron's (v, w) to (0, 0)."""
        def _state(N):
            v = torch.zeros((batch_size, N), dtype=torch.float32, device=device)
            w = torch.zeros((batch_size, N), dtype=torch.float32, device=device)
            return v, w

        self.v1, self.w1 = _state(N_hidden)
        self.v2, self.w2 = _state(N_out)

    def step(self, x_t):
        """One network timestep: propagate one input spike-pattern x_t
        through both layers and return each layer's output spikes."""
        I1 = self.W1(x_t)
        self.v1, self.w1, s1 = bistable_neuron(self.v1, self.w1, I1)

        I2 = self.W2(s1)
        self.v2, self.w2, s2 = bistable_neuron(self.v2, self.w2, I2)

        return s1, s2

    def forward(self, spike_train):
        """
        spike_train : (T, batch, N_in) Poisson-encoded input spikes.

        Runs T stimulus steps, then T_wait silent steps (no input) -- this
        silent tail is what the persistent-memory property is tested
        against: a well-tuned bistable neuron keeps firing through the
        silence instead of falling back to rest.

        Returns a dict with:
          'hidden' : (T+T_wait, batch, N_hidden) hidden-layer spike record
          'output' : (T+T_wait, batch, N_out)    output-layer spike record
          'vote'   : (batch, N_out) spike counts over the last T_eval steps
                     of 'output' -- this is what the loss/accuracy are
                     computed from.
        """
        T = spike_train.shape[0]
        batch_size = spike_train.shape[1]

        self.reset_state(batch_size)

        hidden_record = []
        output_record = []

        # ── phase 1: stimulus ────────────────────────────────────────────
        for t in range(T):
            s1, s2 = self.step(spike_train[t])
            hidden_record.append(s1)
            output_record.append(s2)

        # ── phase 2: silence (memory test) ──────────────────────────────
        if T_wait > 0:
            silence = torch.zeros_like(spike_train[0])
            for t in range(T_wait):
                s1, s2 = self.step(silence)
                hidden_record.append(s1)
                output_record.append(s2)

        hidden_record = torch.stack(hidden_record, dim=0)
        output_record = torch.stack(output_record, dim=0)

        vote = output_record[-T_eval:].sum(dim=0)

        return {'hidden': hidden_record, 'output': output_record, 'vote': vote}


def get_dataloaders(batch_size=batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    train_set = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def poisson_encode(imgs, T):
    """imgs: (batch, 784) pixel values in [0, 1] -> (T, batch, 784) spikes,
    each pixel firing as an independent Bernoulli/Poisson process at a rate
    equal to its own intensity."""
    rates = imgs.unsqueeze(0).expand(T, -1, -1)
    return torch.bernoulli(rates)
