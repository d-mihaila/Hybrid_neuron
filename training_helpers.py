"""
Training and evaluation loop for network_model.Network.

Kept separate from network_model.py so the network's definition (what it
*is*) and its training procedure (how it's *fit*) don't live in the same
file -- makes both easier to read on their own.
"""
import torch
import torch.nn as nn

from params import T, TOTAL_STEPS, EVAL_EVERY, batch_size as _batch_size
from network_model import device, poisson_encode


def evaluate(net, loader, loss_fn, T=T, batch_size=_batch_size):
    """Full pass over `loader` (typically the test set) with gradients off.
    Returns (mean_loss, accuracy)."""
    net.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.view(imgs.shape[0], -1).to(device)
            targets = targets.to(device)

            spike_train = poisson_encode(imgs, T).to(device)
            out = net(spike_train)
            loss_val = loss_fn(out['vote'], targets)

            _, predicted = out['vote'].max(1)
            total_correct += (predicted == targets).sum().item()
            total_loss += loss_val.item() * imgs.shape[0]
            total_samples += imgs.shape[0]

    return total_loss / total_samples, total_correct / total_samples


def train_fixed_steps(net, train_loader, test_loader, optimizer, loss_fn,
                       total_steps=TOTAL_STEPS, eval_every=EVAL_EVERY,
                       T=T, batch_size=_batch_size, verbose=True):
    """
    Train for exactly `total_steps` gradient steps (not epochs), evaluating
    on the full test set every `eval_every` steps.

    total_steps / eval_every default to params.TOTAL_STEPS / params.EVAL_EVERY
    (a full run) -- pass smaller values (as notebook 2 does) for a quick demo.

    Returns a history dict keyed by step number:
        step, train_loss, train_acc  (cheap -- current batch only)
        test_loss,  test_acc         (full test-set evaluate(), every eval_every steps)
    """
    history = {'step': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    def infinite_loader(loader):
        while True:
            yield from loader

    net.train()
    step = 0

    for imgs, targets in infinite_loader(train_loader):
        if step >= total_steps:
            break

        imgs = imgs.view(imgs.shape[0], -1).to(device)
        targets = targets.to(device)
        spike_train = poisson_encode(imgs, T).to(device)

        out = net(spike_train)
        loss_val = loss_fn(out['vote'], targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        _, predicted = out['vote'].max(1)
        batch_acc = (predicted == targets).float().mean().item()
        batch_loss = loss_val.item()

        step += 1

        if step % eval_every == 0 or step == total_steps:
            te_loss, te_acc = evaluate(net, test_loader, loss_fn, T=T, batch_size=batch_size)
            net.train()   # evaluate() leaves the net in eval mode

            history['step'].append(step)
            history['train_loss'].append(batch_loss)
            history['train_acc'].append(batch_acc)
            history['test_loss'].append(te_loss)
            history['test_acc'].append(te_acc)

            if verbose:
                print(f"step {step:5d}/{total_steps}  "
                      f"train loss {batch_loss:.4f}  train acc {batch_acc*100:.1f}%  "
                      f"test loss {te_loss:.4f}  test acc {te_acc*100:.1f}%")

    net.eval()
    return history
