"""Zeroth-order SPSA, MeZO flavour: nudge every weight by +eps*z and -eps*z,
look at how the loss changed, step along z. Two forward passes, no autograd,
no activation memory. Built for fine-tuning; from scratch it's a long shot.
"""
import torch

from . import Rule, register


@register
class MeZO(Rule):
    name = "mezo"
    autograd = "none"
    blurb = "Two-point SPSA: perturb all weights, compare two losses, step. Zero autograd, zero activation memory."
    paper = "Spall 1992 (SPSA); Malladi et al. 2023 (MeZO)"
    defaults = dict(lr=1e-4, eps=1e-3)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.params = [p for p in model.parameters()]

    @torch.no_grad()
    def step(self, x, y):
        eps, lr = self.hp["eps"], self.hp["lr"]
        z = [torch.randn_like(p) for p in self.params]
        for p, zi in zip(self.params, z):
            p.add_(zi, alpha=eps)
        lp = self.model(x, y)[1]
        for p, zi in zip(self.params, z):
            p.add_(zi, alpha=-2 * eps)
        lm = self.model(x, y)[1]
        g = ((lp - lm) / (2 * eps)).item()
        for p, zi in zip(self.params, z):
            p.add_(zi, alpha=eps - lr * g)  # restore, then step
        self.forwards += 2
        return 0.5 * (lp.item() + lm.item())
