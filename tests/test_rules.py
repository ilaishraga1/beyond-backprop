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


def test_census_heads_update_is_the_local_attention_gradient():
    """census_heads' hand-written Q/K update == autograd of CE(EM posterior, attention), prior held fixed."""
    from bbp.rules.census_heads import CensusHeads
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    m = GPT(cfg)
    r = CensusHeads(m, max_offset=3)
    x, y = batch(cfg)
    for _ in range(3):
        r.step(x, y)
    r._refresh()
    with torch.no_grad():
        m(x)
    r._teach_heads(x, y)
    W = m.blocks[0].attn.qkv.weight
    manual = W.grad.clone()
    z, b = r.caught["z"].detach(), m.blocks[0].attn.qkv.bias.detach()
    W2 = W.detach().clone().requires_grad_(True)
    B, T, C, hd = x.size(0), cfg.block_size, cfg.n_embd, r.hd
    ar = torch.arange(T)
    offs = ar[:, None] - ar[None, :]
    L = torch.full((B, T, T), float("-inf"))
    L[:, ar, ar] = 0.0
    for d in range(1, 4):
        t = ar[d:]
        L[:, t, t - d] = r.table[d - 1][x[:, :-d], x[:, d:], y[:, d:]]
    loss = 0.0
    for h in range(r.nh):
        q = z @ W2[h * hd:(h + 1) * hd].T + b[h * hd:(h + 1) * hd]
        k = z @ W2[C + h * hd:C + (h + 1) * hd].T + b[C + h * hd:C + (h + 1) * hd]
        S = ((q @ k.transpose(1, 2)) / math.sqrt(hd)).masked_fill(offs < 0, float("-inf"))
        a = S.softmax(-1)
        prior = (a.detach() + 1e-9).log()
        tgt = (prior + L.masked_fill((offs >= 1) & (offs <= h), float("-inf"))).softmax(-1)
        loss = loss - (tgt * S.log_softmax(-1).masked_fill(offs < 0, 0)).sum() / (B * T)
    loss.backward()
    assert torch.allclose(W2.grad, manual, atol=1e-6, rtol=1e-4)
