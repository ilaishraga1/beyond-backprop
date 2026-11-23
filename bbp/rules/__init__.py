"""Learning-rule registry.

Drop a file in this folder that defines a subclass of `Rule` decorated with
`@register`, and the arena picks it up automatically. See `_template.py`.
"""
from __future__ import annotations

import importlib
import pkgutil

import torch

RULES: dict[str, type["Rule"]] = {}

# How much the rule leans on autograd. The arena enforces "none" (any call to
# .backward() or torch.autograd.grad raises), the others are honesty labels
# shown on the leaderboard.
AUTOGRAD_LEVELS = {
    "none": "no backward pass at all",
    "local": "backprop only inside a layer/block, never end to end",
    "modified": "end-to-end backward pass, but not the true gradient",
    "full": "plain backprop",
}


class Rule:
    """Base class. A rule owns everything about how the model learns.

    Subclasses set:
      name      registry key, also the CLI name
      autograd  one of AUTOGRAD_LEVELS
      blurb     one line for the leaderboard
      paper     where the idea comes from (or "" if it's yours)
      defaults  hyperparameters, overridable with --set key=value
    and implement step(x, y) -> float (the training loss on this batch).

    Bump self.forwards by the number of full-model forward passes over the
    batch the step cost (a jvp or a vmap'd population counts what it computes),
    so the leaderboard can compare compute, not just wall-clock.
    """

    name: str = ""
    autograd: str = "none"
    blurb: str = ""
    paper: str = ""
    defaults: dict = {}
    tune: str = "lr"  # the one hyperparameter the tournament sweeps (x1/3, x1, x3)

    def __init__(self, model: torch.nn.Module, **hp):
        self.model = model
        self.hp = {**self.defaults, **hp}
        self.forwards = 0

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:  # pragma: no cover
        raise NotImplementedError


def register(cls):
    assert cls.name, f"{cls.__name__} needs a name"
    assert cls.autograd in AUTOGRAD_LEVELS, f"{cls.name}: autograd must be one of {list(AUTOGRAD_LEVELS)}"
    assert cls.name not in RULES, f"duplicate rule name {cls.name}"
    RULES[cls.name] = cls
    return cls


def _discover():
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_discover()
