"""SHARPNESS-AWARE: don't follow the slope where you stand, follow the slope where a
small step uphill would land you. Minima that stay low after that nudge are wide, and
wide minima generalise. The update is the gradient at a perturbed point, not at the
current weights, so 'modified'.

mode="exact": two backward passes per step (ascent, then descent): Foret et al. 2021.
mode="lagged": reuse the previous step's gradient as the uphill direction, so it costs
one backward pass, the same as backprop (in the spirit of Du et al. 2022, SAF).
"""
import torch

from . import register
from .backprop import Backprop


@register
class SAM(Backprop):
    name = "sam"
    autograd = "modified"
    blurb = "Each update uses the gradient taken a small step uphill from the current weights, which steers training toward wide minima."
    paper = "Foret et al. 2021 (SAM); lagged variant after Du et al. 2022"
    defaults = dict(Backprop.defaults, rho=0.15, mode="exact")  # rho picked on a 2-min val screen (0.05 vs 0.15)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.prev = None  # lagged mode: last step's gradient

    def _grads(self):
        return [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]

    def _nudge(self, g):
        """Move the weights rho uphill along g; return the offsets so they can be undone."""
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t) for t in g]))
        e = torch._foreach_mul(g, (self.hp["rho"] / (norm + 1e-12)).item())
        torch._foreach_add_(self.params, e)
        return e

    def step(self, x, y):
        from ._regularise import regularise
        x = regularise(self, [self.opt], x)
        e = None
        if self.hp["mode"] == "exact":
            self.opt.zero_grad(set_to_none=True)
            _, loss = self.model(x, y)
            loss.backward()
            with torch.no_grad():
                e = self._nudge(self._grads())
            self.forwards += 1
        elif self.prev is not None:
            with torch.no_grad():
                e = self._nudge(self.prev)
        self.opt.zero_grad(set_to_none=True)
        _, l2 = self.model(x, y)
        l2.backward()
        with torch.no_grad():
            if e is not None:
                torch._foreach_sub_(self.params, e)
            if self.hp["mode"] == "lagged":
                self.prev = self._grads()  # zero_grad(set_to_none) leaves these alone
        self.opt.step()
        self.forwards += 1
        return (loss if self.hp["mode"] == "exact" else l2).item()
