"""MIRROR NEURONS: the backward weights watch the forward ones and learn the same lesson.

Backprop needs every backward synapse to be an exact copy of a forward one (the
'weight transport problem'). Feedback alignment uses fixed random backward weights
instead and pays for it. Here the backward weights are separate and start random,
but they are *plastic*: each receives the same Hebbian product (error x activity) that
its forward twin receives, since both synapses touch the same two neurons. Nobody ever
copies a weight, yet two strangers who live the same life end up alike
(Kolen & Pollack 1994; Akrout et al. 2019).

mirror_neurons is the whole transformer: attention, MLPs and readout all learn.
Still an end-to-end backward pass, so 'modified'.

Ablation: mirror=False freezes the backward weights, which is feedback_alignment.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from .feedback_alignment import FALinear, swap_linears


@register
class MirrorNeurons(Rule):
    name = "mirror_neurons"
    autograd = "modified"
    blurb = "Separate, random-born backward weights that receive the same Hebbian update as their forward twins, so they drift into alignment without any weight being copied."
    paper = "Kolen & Pollack 1994; Akrout et al. 2019 (weight mirrors)"
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True, half=0, noise=0.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        kw = dict(lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        self.opt = torch.optim.AdamW(model.parameters(), **kw)
        self.opt_back = torch.optim.AdamW([m.feedback for m in self.twins], **kw)

    def step(self, x, y):
        x = regularise(self, [self.opt, self.opt_back], x)
        _, loss = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.hp["mirror"]:
            for m in self.twins:
                m.feedback.grad = m.weight.grad.clone()
            self.opt_back.step()
        self.opt.step()
        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        """Mean cosine between backward weights and the forward weights they stand in for."""
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item() for m in self.twins) / len(self.twins)
