"""No-cheating guard for rules that claim autograd = "none"."""
from __future__ import annotations

import contextlib

import torch


class BackpropDetected(RuntimeError):
    pass


@contextlib.contextmanager
def no_backprop(rule_name: str):
    """Make every reverse-mode entry point raise while the block runs."""
    def boom(*_a, **_k):
        raise BackpropDetected(
            f"rule '{rule_name}' declares autograd='none' but called a backward pass. "
            "Either remove it or change the rule's autograd level."
        )

    saved = (torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad)
    torch.Tensor.backward = boom
    torch.autograd.backward = boom
    torch.autograd.grad = boom
    try:
        yield
    finally:
        torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad = saved
