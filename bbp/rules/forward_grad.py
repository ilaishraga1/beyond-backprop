"""Forward gradient: pick a random direction v, get the directional derivative
(grad . v) for free in the forward pass with forward-mode AD, and use
(grad . v) * v as an unbiased estimate of the gradient. No backward pass.
"""
import torch
from torch.func import jvp

from . import Rule, register
from ._util import functional_loss, params_of, set_grads


@register
class ForwardGradient(Rule):
    name = "forward_grad"
    autograd = "none"
    blurb = "Forward-mode JVP along a random direction, (g.v)v as the gradient. Unbiased, very noisy."
    paper = "Baydin et al. 2022, 'Gradients without Backpropagation'"
    defaults = dict(lr=2e-4, tangents=1)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.opt = torch.optim.Adam(model.parameters(), lr=self.hp["lr"])

    def step(self, x, y):
        params = params_of(self.model)
        est = {k: torch.zeros_like(p) for k, p in params.items()}
        k = self.hp["tangents"]
        for _ in range(k):
            v = {n: torch.randn_like(p) for n, p in params.items()}
            loss, dd = jvp(lambda p: functional_loss(self.model, p, x, y), (params,), (v,))
            for n in est:
                est[n].add_(v[n], alpha=dd.item() / k)
        set_grads(self.model, est)
        self.opt.step()
        self.forwards += k
        return loss.item()
