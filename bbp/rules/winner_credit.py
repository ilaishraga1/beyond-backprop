"""WINNER CREDIT: backprop, except inside every MLP only the k most strongly driven hidden
neurons (per token) receive any error. The forward pass is untouched; the rest of the layer
fired but learns nothing from this token. Competitive, winner-take-all plasticity.
Not the true gradient, so 'modified'."""
import torch

from . import register
from .backprop import Backprop


class _Winners(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, k):
        idx = h.abs().topk(k, dim=-1).indices
        mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(mask)
        return h.view_as(h)

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return g * mask, None


@register
class WinnerCredit(Backprop):
    name = "winner_credit"
    autograd = "modified"
    blurb = "Backprop where, in every MLP, only the k most strongly driven hidden neurons per token get any error signal."
    paper = "k-winners-take-all plasticity (e.g. Makhzani & Frey 2015; Ahmad & Scheinkman 2019)"
    defaults = dict(Backprop.defaults, k=64)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        k = int(self.hp["k"])
        for blk in model.blocks:
            blk.mlp.fc.register_forward_hook(lambda m, i, out: _Winners.apply(out, k) if torch.is_grad_enabled() else out)
