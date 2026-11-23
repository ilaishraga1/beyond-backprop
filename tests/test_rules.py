import math

import pytest
import torch

from bbp.arena import run
from bbp.guard import BackpropDetected, no_backprop
from bbp.model import GPT, GPTConfig
from bbp.rules import RULES, Rule, register

TINY = dict(vocab_size=11, block_size=16, n_layer=2, n_head=2, n_embd=32)


def batch(cfg, B=4):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(cfg.vocab_size, (B, cfg.block_size), generator=g)
    return x, torch.roll(x, -1, 1)


@pytest.mark.parametrize("name", sorted(RULES))
def test_rule_steps_and_learns_something(name):
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    model = GPT(cfg)
    rule = RULES[name](model)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    x, y = batch(cfg)
    for _ in range(4):
        if rule.autograd == "none":
            with no_backprop(name):
                loss = rule.step(x, y)
        else:
            loss = rule.step(x, y)
        assert isinstance(loss, float) and math.isfinite(loss)
    assert rule.forwards >= 4
    changed = any(not torch.equal(before[k], v) for k, v in model.named_parameters())
    if name != "hill_climb":  # hill climbing may legitimately reject every jiggle
        assert changed, f"{name} never touched the weights"
    # the model's own forward still works after the rule rewired it
    _, l = model(x, y)
    assert math.isfinite(l.item())


def test_guard_catches_a_cheater():
    class Cheater(Rule):
        name, autograd = "cheater", "none"

        def step(self, x, y):
            self.model(x, y)[1].backward()
            return 0.0

    cfg = GPTConfig(**TINY)
    r = Cheater(GPT(cfg))
    with pytest.raises(BackpropDetected):
        with no_backprop("cheater"):
            r.step(*batch(cfg))
    # and the guard un-patches itself
    t = torch.ones(1, requires_grad=True)
    (t * 2).sum().backward()
    assert t.grad.item() == 2.0


def test_guard_catches_autograd_grad():
    with pytest.raises(BackpropDetected):
        with no_backprop("x"):
            t = torch.ones(1, requires_grad=True)
            torch.autograd.grad((t * 2).sum(), t)


def test_arena_end_to_end_cpu():
    res = run("backprop", minutes=1, device="cpu", hp={}, cfg=GPTConfig(block_size=16, n_layer=1, n_embd=32, n_head=2),
              batch_size=4, max_steps=20, save=False, quiet=True)
    assert res["steps"] == 20 and math.isfinite(res["test_loss"]) and len(res["sample"]) > 300
    assert res["test_loss"] < res["init_val"]


def test_every_rule_is_honestly_labelled():
    for name, cls in RULES.items():
        assert cls.blurb and cls.autograd
