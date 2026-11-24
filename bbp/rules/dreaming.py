"""DREAMING: the model learns by dreaming.

Awake, it reads real Shakespeare. Asleep, it reads its own dream: the same
passage with every character replaced by the one the model *guessed* would
come there (sampled from its own predictions). Then, in every layer:

    strengthen what fired while awake,   weaken what fired while dreaming
    dW  =  post_awake (x) pre_awake   -   post_dream (x) pre_dream

That's wake/sleep Hebbian learning, the positive/negative phases of a
Boltzmann machine. For a linear layer, post (x) pre is the gradient of its
"goodness" ||Wx||^2 / 2, so this is Forward-Forward with the model's own
dreams as the negative data (which Hinton suggested), done with plain Hebb
instead of per-layer backprop.

The readout learns the same way: wire up (features, true next char) while
awake and un-wire (features, dreamt next char) while asleep. In expectation
that's exactly the cross-entropy gradient for the last layer (the delta rule),
so the readout gets an honest signal and everything below it gets dreams.
Nothing runs backward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Rule, register


@register
class Dreaming(Rule):
    name = "dreaming"
    autograd = "none"
    blurb = "Wake/sleep Hebbian: strengthen what fires on real text, weaken what fires on the model's own dreams. Readout: real-vs-dreamt delta rule."
    paper = "Hinton & Sejnowski 1983 (Boltzmann machines); Crick & Mitchison 1983 (we dream to forget); Hinton 2022 (FF with model negatives)"
    defaults = dict(lr=3e-4, weight_decay=0.01, temperature=1.0, dream_hidden=True)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.hidden = [m for b in model.blocks for m in b.modules() if isinstance(m, nn.Linear)]
        self.acts = {}
        self.record = None
        for m in self.hidden + [model.lm_head]:
            m.register_forward_hook(self._hook)
        params = [p for m in self.hidden + [model.lm_head] for p in m.parameters()]
        self.opt = torch.optim.AdamW(params, lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])

    def _hook(self, mod, inp, out):
        if self.record is not None:
            self.acts[(self.record, mod)] = (inp[0], out)

    def _hebb(self, phase, m, N):
        x, y = self.acts[(phase, m)]
        return torch.einsum("bto,bti->oi", y, x) / N, y.sum((0, 1)) / N

    @torch.no_grad()
    def step(self, x, y):
        N = x.numel()
        V = self.model.cfg.vocab_size
        self.record = "wake"
        logits, loss = self.model(x, y)
        p = (logits / self.hp["temperature"]).softmax(-1)
        guess = torch.multinomial(p.reshape(-1, V), 1).view_as(y)  # what the model thought came next
        dream = torch.cat([x[:, :1], guess[:, :-1]], dim=1)  # the passage as the model imagines it
        self.record = "dream"
        self.model(dream)
        self.record = None

        for m in (self.hidden if self.hp["dream_hidden"] else []):
            ww, bw = self._hebb("wake", m, N)
            wd, bd = self._hebb("dream", m, N)
            m.weight.grad = -(ww - wd)  # optimizers descend, Hebb ascends
            m.bias.grad = -(bw - bd)
        # readout: (features, real next char) minus (features, dreamt next char) = delta rule
        h, _ = self.acts[("wake", self.model.lm_head)]
        target = F.one_hot(y, V).float() - F.one_hot(guess, V).float()
        self.model.lm_head.weight.grad = -torch.einsum("btv,btc->vc", target, h) / N

        self.opt.step()
        self.acts.clear()
        self.forwards += 2
        return loss.item()
