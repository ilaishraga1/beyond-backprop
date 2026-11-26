"""Backdrop: the forward pass is untouched, but on the way back the error signal at the
output of each block (except the last) is dropped for a random fraction `p` of token
positions and the rest are scaled up by 1/(1-p). Lower layers therefore learn from a
different random subset of the sequence each step. Golkar & Cranmer 2018."""
import torch


class _GradMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        keep = (torch.rand(g.shape[:-1] + (1,), device=g.device) >= ctx.p).to(g.dtype)
        return g * keep / (1 - ctx.p), None


def install_backdrop(model, p):
    """Mask the backward signal leaving every block but the last. No-op for p == 0."""
    if not p:
        return
    for blk in list(model.blocks)[:-1]:
        blk.register_forward_hook(lambda m, i, out: _GradMask.apply(out, p) if torch.is_grad_enabled() else out)
