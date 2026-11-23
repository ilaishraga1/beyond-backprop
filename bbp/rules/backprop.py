"""The thing we're trying to beat."""
import torch

from . import Rule, register


@register
class Backprop(Rule):
    name = "backprop"
    autograd = "full"
    blurb = "Plain backprop + AdamW. The baseline everyone loses to."
    paper = "Rumelhart, Hinton & Williams 1986"
    defaults = dict(lr=2e-3, weight_decay=0.01)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])

    def step(self, x, y):
        _, loss = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.forwards += 1
        return loss.item()
