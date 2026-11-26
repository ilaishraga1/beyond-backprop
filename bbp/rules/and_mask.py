"""HARD TO VARY (AND-mask): split each batch into `groups` slices of random windows, get each
slice's gradient, and let a weight move only where at least a fraction `tau` of the slices
agree on which way. Patterns that only some windows show (one play's names and plot) disagree
and are masked out; patterns every slice shows (spelling, grammar) agree and are learned.
The update is the masked mean gradient, not the gradient, so 'modified'.
Parascandolo et al. 2020, 'Learning explanations that are hard to vary'.
"""
import torch
from torch.func import functional_call, grad_and_value, vmap

from . import register
from ._regularise import regularise
from .backprop import Backprop


@register
class AndMask(Backprop):
    name = "and_mask"
    autograd = "modified"
    blurb = "Each weight moves only where most slices of the batch agree on the sign of its gradient: learn what is hard to vary."
    paper = "Parascandolo et al. 2020, 'Learning explanations that are hard to vary'"
    defaults = dict(Backprop.defaults, groups=4, tau=1.0)

    def step(self, x, y):
        T = getattr(self, "teacher", None)
        if self.hp.get("teach") and (T is None or not T.ready):
            return super().step(x, y)  # counting phase of the optional teacher: plain step
        x0, x = x, regularise(self, [self.opt], x)
        G = int(self.hp["groups"])
        B = x.shape[0] // G * G
        a = self.hp.get("teach", 0.0)
        p = T(x0, y) if a else torch.zeros(x.shape + (self.model.cfg.vocab_size,), device=x.device)
        names = [n for n, _ in self.model.named_parameters()]
        params = {n: q.detach() for n, q in self.model.named_parameters()}
        bufs = dict(self.model.named_buffers())

        def lossf(prm, xb, yb, pb):
            logits, _ = functional_call(self.model, (prm, bufs), (xb,))
            lp = logits.log_softmax(-1)
            ce = -lp.gather(-1, yb[..., None]).mean()
            return (1 - a) * ce + a * -(pb * lp).sum(-1).mean() if a else ce

        sh = lambda t: t[:B].reshape(G, B // G, *t.shape[1:])
        gs, losses = vmap(grad_and_value(lossf), in_dims=(None, 0, 0, 0))(params, sh(x), sh(y), sh(p))
        tau = self.hp["tau"]
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, q in zip(names, self.model.parameters()):
                g = gs[n]
                agree = g.sign().mean(0).abs() >= tau - 1e-6
                q.grad = g.mean(0) * agree
        self.opt.step()
        self.forwards += 1
        return losses.mean().item()
