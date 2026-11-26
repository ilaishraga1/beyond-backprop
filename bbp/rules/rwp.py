"""RANDOM WEIGHT PERTURBATION: every gradient is taken at a freshly jittered copy of the weights
(each weight nudged by Gaussian noise of sigma x the RMS of its own row), then the jitter is
removed and the clean weights step. One backward pass, like backprop; the update is the gradient
of a noise-smoothed loss at a random point, not at the weights, so 'modified'.
Li et al. 2023, 'Revisiting random weight perturbation for efficiently improving generalization'."""
import torch

from . import register
from .backprop import Backprop


@register
class RWP(Backprop):
    name = "rwp"
    autograd = "modified"
    blurb = "Each gradient is taken at a randomly jittered copy of the weights (noise scaled to each row), then the clean weights step."
    paper = "Li et al. 2023 (random weight perturbation)"
    defaults = dict(Backprop.defaults, sigma=0.01)

    def step(self, x, y):
        ps = [p for p in self.model.parameters() if p.dim() == 2]
        with torch.no_grad():
            eps = [torch.randn_like(p) * p.pow(2).mean(1, keepdim=True).sqrt() * self.hp["sigma"] for p in ps]
            torch._foreach_add_(ps, eps)
        self._eps = (ps, eps)
        return super().step(x, y)

    def _before_opt(self):
        ps, eps = self._eps
        with torch.no_grad():
            torch._foreach_sub_(ps, eps)
