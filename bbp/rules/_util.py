"""Helpers shared by the gradient-free rules."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call


def params_of(model) -> dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in model.named_parameters()}


def functional_loss(model, params: dict, x, y) -> torch.Tensor:
    logits, _ = functional_call(model, params, (x,))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def set_grads(model, grads: dict[str, torch.Tensor]):
    """Hand a home-made gradient estimate to a normal torch optimizer."""
    for k, p in model.named_parameters():
        p.grad = grads[k].to(p.dtype)


@torch.no_grad()
def loss_now(model, x, y) -> float:
    return model(x, y)[1].item()
