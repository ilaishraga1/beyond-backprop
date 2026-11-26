"""The thing we're trying to beat."""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach


@register
class Backprop(Rule):
    name = "backprop"
    autograd = "full"
    blurb = "Plain backprop + AdamW. The baseline everyone loses to."
    paper = "Rumelhart, Hinton & Williams 1986"
    defaults = dict(lr=2e-3, weight_decay=0.01, half=0, noise=0.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)  # optional dreamed text, off by default
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)  # optional counting teacher, off by default
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.forwards += 1
        return loss.item()
