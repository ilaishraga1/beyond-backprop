"""Sanity floor: freeze the random network, train only the readout.

If your fancy rule can't beat a random reservoir with a trained linear head,
the deep part of the network isn't learning anything.
"""
import torch

from . import Rule, register


@register
class Reservoir(Rule):
    name = "reservoir"
    autograd = "local"
    blurb = "Random frozen transformer, only the final readout learns. The 'is depth doing anything?' floor."
    paper = "Jaeger 2001 (echo state networks)"
    defaults = dict(lr=3e-3)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.trainable = list(model.ln_f.parameters()) + list(model.lm_head.parameters())
        for p in model.parameters():
            p.requires_grad_(False)
        for p in self.trainable:
            p.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.trainable, lr=self.hp["lr"])

    def step(self, x, y):
        _, loss = self.model(x, y)  # grads only reach ln_f + lm_head
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.forwards += 1
        return loss.item()
