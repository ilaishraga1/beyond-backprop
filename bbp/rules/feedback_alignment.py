"""Feedback alignment and sign symmetry: keep the backward pass, but stop it
from using the forward weights (the 'weight transport problem' brains can't
solve). Every nn.Linear is swapped for one whose backward sends the error
through a fixed random matrix B (FA) or through sign(W) (sign symmetry).
Weight updates themselves are still the usual local outer product.
"""
import torch
import torch.nn as nn

from . import Rule, register
from ._teacher import teach


class _FALinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, feedback):
        ctx.save_for_backward(x, weight, feedback)
        ctx.has_bias = bias is not None
        out = x @ weight.t()
        return out + bias if bias is not None else out

    @staticmethod
    def backward(ctx, g):
        x, weight, feedback = ctx.saved_tensors
        B = (torch.sign(weight) * weight.abs().mean()).detach() if feedback is None else feedback
        gx = g @ B  # <- the whole trick: not W
        gw = g.reshape(-1, g.size(-1)).t() @ x.reshape(-1, x.size(-1))
        gb = g.reshape(-1, g.size(-1)).sum(0) if ctx.has_bias else None
        return gx, gw, gb, None


class FALinear(nn.Module):
    def __init__(self, lin: nn.Linear, mode: str, scale: float):
        super().__init__()
        self.weight, self.bias = lin.weight, lin.bias
        self.mode = mode
        if mode == "random":
            w = lin.weight.detach()
            fb = torch.randn_like(w) * w.std().clamp_min(1e-3) * scale
            self.register_buffer("feedback", fb)
        else:
            self.feedback = None

    def forward(self, x):
        return _FALinearFn.apply(x, self.weight, self.bias, self.feedback)


def swap_linears(module: nn.Module, mode: str, scale: float):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, FALinear(child, mode, scale))
        else:
            swap_linears(child, mode, scale)


class _FABase(Rule):
    autograd = "modified"
    mode = "random"
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, teach=0.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, self.mode, self.hp["scale"])
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])

    def step(self, x, y):
        logits, loss = self.model(x, y)
        loss = teach(self, x, y, logits, loss)  # optional counting teacher, off by default
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.forwards += 1
        return loss.item()


@register
class FeedbackAlignment(_FABase):
    name = "feedback_alignment"
    mode = "random"
    blurb = "Backward pass through fixed random matrices instead of W^T. The forward weights learn to 'align'."
    paper = "Lillicrap et al. 2016, 'Random synaptic feedback weights support error backpropagation'"


@register
class SignSymmetry(_FABase):
    name = "sign_symmetry"
    mode = "sign"
    blurb = "Backward pass through sign(W) only: the feedback knows the signs, not the magnitudes."
    paper = "Liao, Leibo & Poggio 2016, 'How important is weight symmetry in backpropagation?'"
