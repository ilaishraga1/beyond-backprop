"""NOISY BACKPROP: the forward pass is exact, and so is every weight's local outer product,
but the error signal arriving at each block boundary on the way down is jittered with fresh
Gaussian noise (sigma x its own RMS). Like a backward pathway made of unreliable neurons.
Not the true gradient, so 'modified'. The control is plain backprop."""
import torch

from . import register
from .backprop import Backprop


class _Jitter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sigma):
        ctx.sigma = sigma
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g + ctx.sigma * g.pow(2).mean().sqrt() * torch.randn_like(g), None


@register
class NoisyBackprop(Backprop):
    name = "noisy_backprop"
    autograd = "modified"
    blurb = "Backprop whose error signal gets fresh noise (sigma x its RMS) every time it crosses a block boundary on the way down."
    paper = "gradient-noise injection, e.g. Neelakantan et al. 2015 (here in the backward pass, per layer)"
    defaults = dict(Backprop.defaults, sigma=0.5, where="blocks")

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        s = self.hp["sigma"]
        hook = lambda m, i, out: _Jitter.apply(out, s) if torch.is_grad_enabled() else out
        if s:
            if self.hp["where"] == "blocks":  # error leaving every block but the last
                for blk in list(model.blocks)[:-1]:
                    blk.register_forward_hook(hook)
            elif self.hp["where"] == "attn":  # only the error entering each attention branch (long-range context)
                for blk in model.blocks:
                    blk.attn.register_forward_hook(hook)
            elif self.hp["where"] == "mlp":  # only the error entering each MLP branch (local context)
                for blk in model.blocks:
                    blk.mlp.register_forward_hook(hook)
