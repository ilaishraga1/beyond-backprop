"""The dumbest possible learner: jiggle every weight at random, keep the
jiggle if the loss went down, otherwise undo it. Step size adapts with the
1/5th success rule. A floor for the gradient-free crowd.
"""
import torch

from . import Rule, register


@register
class HillClimb(Rule):
    name = "hill_climb"
    autograd = "none"
    blurb = "(1+1)-ES: random jiggle, keep it if the loss dropped. Adaptive step size. Pure dumb luck, as a floor."
    paper = "Rechenberg 1973 (1/5th success rule)"
    defaults = dict(sigma=0.01, adapt=1.1)
    tune = "sigma"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.params = list(model.parameters())
        self.sigma = self.hp["sigma"]

    @torch.no_grad()
    def step(self, x, y):
        before = self.model(x, y)[1].item()
        z = [torch.randn_like(p) * self.sigma for p in self.params]
        for p, zi in zip(self.params, z):
            p.add_(zi)
        after = self.model(x, y)[1].item()
        a = self.hp["adapt"]
        if after < before:
            self.sigma *= a ** 4  # success: be bolder (1/5th rule: x a^4 on success, / a on failure)
        else:
            for p, zi in zip(self.params, z):
                p.sub_(zi)
            self.sigma /= a
        self.forwards += 2
        return min(before, after)
