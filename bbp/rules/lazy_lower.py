"""LAZY LOWER LAYERS: the error only reaches the bottom `cut` blocks (and the embeddings) on
every `every`-th step; on the other steps it stops at block `cut`, so the backward pass is
shorter and the lower layers learn more slowly than the top. Not the true gradient, so 'modified'."""
import torch

from . import register
from .backprop import Backprop


@register
class LazyLower(Backprop):
    name = "lazy_lower"
    autograd = "modified"
    blurb = "The error reaches the bottom blocks only every k-th step; otherwise it stops halfway down."
    paper = "cf. FreezeOut (Brock et al. 2017)"
    defaults = dict(Backprop.defaults, cut=2, every=4)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.k = 0
        blk = model.blocks[min(int(self.hp["cut"]), len(model.blocks) - 1)]  # tiny test models have 2 blocks
        blk.register_forward_pre_hook(lambda m, args: (args[0].detach(),) if self._lazy else None)
        self._lazy = False

    def step(self, x, y):
        self._lazy = self.k % int(self.hp["every"]) != 0
        self.k += 1
        try:
            return super().step(x, y)
        finally:
            self._lazy = False
