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
from ._backdrop import install_backdrop
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class MirrorNeurons(Rule):
    name = "mirror_neurons"
    autograd = "modified"
    blurb = "Separate, random-born backward weights that receive the same Hebbian update as their forward twins, so they drift into alignment without any weight being copied."
    paper = "Kolen & Pollack 1994; Akrout et al. 2019 (weight mirrors)"
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True, half=0, noise=0.0, bdrop=0.0, fnoise=0.0, fdrop=0.0, born="random")

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        if self.hp["born"] == "copy":  # ablation: feedback born as an exact copy -> exact backprop credit
            for m in model.modules():
                if isinstance(m, FALinear):
                    m.feedback.copy_(m.weight.detach())
        install_backdrop(model, self.hp["bdrop"])  # optional, off by default
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        kw = dict(lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        self.opt = torch.optim.AdamW(model.parameters(), **kw)
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        jolt = self._jolt()  # optional: unreliable backward synapses for this one step (B is only read going back)
        xa, ya = add_dreams(self, x, y)  # optional dreamed text, off by default
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)  # optional counting teacher, off by default
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if jolt is not None:
                torch._foreach_sub_(self.B, jolt)
            before = torch._foreach_mul(self.W, 1.0) if self.hp["mirror"] else None
            self.opt.step()
            if self.hp["mirror"]:
                # Each backward weight gets exactly the AdamW update its forward twin got
                # (same gradient -> same Adam moments -> same step), with its own decay:
                #   B <- B(1 - lr*wd) - lr*u  ==  B + dW - lr*wd*(B - W_before)
                lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
                gap = torch._foreach_sub(self.B, before)
                torch._foreach_add_(self.B, torch._foreach_sub(self.W, before))
                torch._foreach_add_(self.B, gap, alpha=-lr_wd)
        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def _jolt(self):
        """fnoise: add fresh Gaussian noise (relative to each B's std) to the backward weights;
        fdrop: drop a fraction of backward synapses (rest scaled up). Undone after the backward."""
        f, d = self.hp["fnoise"], self.hp["fdrop"]
        if not (f or d):
            return None
        jolt = []
        for b in self.B:
            j = torch.zeros_like(b)
            if f:
                j += torch.randn_like(b) * b.std() * f
            if d:
                keep = (torch.rand_like(b) >= d).to(b.dtype)
                j += (b + j) * (keep / (1 - d) - 1)
            jolt.append(j)
        torch._foreach_add_(self.B, jolt)
        return jolt

    @torch.no_grad()
    def alignment(self):
        """Mean cosine between backward weights and the forward weights they stand in for."""
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item() for m in self.twins) / len(self.twins)
