"""Train one learning rule on the mini LLM for a fixed wall-clock budget.

    python -m bbp.arena --rule backprop --minutes 5
    python -m bbp.arena --rule mezo --set lr=1e-4 eps=1e-3
    python -m bbp.arena --list

Only time spent inside rule.step() counts against the budget; evaluation is
free. Results go to results/<rule>.json (or --tag to keep variants apart).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import time
from pathlib import Path

import torch

from .data import CharData
from .guard import no_backprop
from .model import GPT, GPTConfig
from .rules import AUTOGRAD_LEVELS, RULES

RESULTS = Path(__file__).resolve().parent.parent / "results"


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_sets(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model, data, batches, batch_size, seed=1234) -> float:
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)  # same val batches every time
    tot = 0.0
    for _ in range(batches):
        x, y = data.batch("val", batch_size, g)
        tot += model(x, y)[1].item()
    model.train(was_training)
    return tot / batches


def run(rule_name: str, minutes: float, device: str, hp: dict, cfg: GPTConfig,
        batch_size: int = 32, eval_every_s: float = 15.0, seed: int = 0, tag: str = "",
        max_steps: int | None = None, save: bool = True, quiet: bool = False) -> dict:
    torch.manual_seed(seed)
    data = CharData(cfg.block_size, device)
    cfg.vocab_size = data.vocab_size
    model = GPT(cfg).to(device)
    rule = RULES[rule_name](model, **hp)
    log = print if not quiet else (lambda *a, **k: None)
    log(f"[{rule_name}] {model.n_params()/1e6:.2f}M params on {device}; autograd={rule.autograd}; hp={rule.hp}")

    guard = (lambda: no_backprop(rule_name)) if rule.autograd == "none" else contextlib.nullcontext
    budget = minutes * 60
    train_time, step, last_eval, diverged = 0.0, 0, -1e9, False
    curve = []
    init_val = evaluate(model, data, 10, batch_size)
    curve.append(dict(t=0.0, step=0, forwards=0, val=init_val))
    g = torch.Generator().manual_seed(seed)
    while train_time < budget and (max_steps is None or step < max_steps):
        x, y = data.batch("train", batch_size, g)
        t0 = time.perf_counter()
        with guard():
            train_loss = rule.step(x, y)  # returns a python float -> device is synced
        train_time += time.perf_counter() - t0
        step += 1
        if not math.isfinite(train_loss):
            diverged = True
            log(f"[{rule_name}] diverged at step {step}")
            break
        if train_time - last_eval >= eval_every_s:
            last_eval = train_time
            v = evaluate(model, data, 10, batch_size)
            curve.append(dict(t=round(train_time, 2), step=step, forwards=rule.forwards, val=v))
            log(f"[{rule_name}] t={train_time:6.1f}s step={step:6d} train={train_loss:.3f} val={v:.3f}")

    final = evaluate(model, data, 50, batch_size) if not diverged else float("nan")
    curve.append(dict(t=round(train_time, 2), step=step, forwards=rule.forwards, val=final))
    prompt = torch.tensor([data.encode("ROMEO:\n")], device=device)
    sample = data.decode(model.generate(prompt, 300, generator=torch.Generator().manual_seed(0))[0].tolist())
    log(f"[{rule_name}] final val loss {final:.4f} after {step} steps / {train_time:.0f}s\n--- sample ---\n{sample}\n")

    result = dict(
        rule=rule_name, tag=tag, autograd=rule.autograd, blurb=rule.blurb, paper=rule.paper,
        hp=rule.hp, val_loss=final, init_val=init_val, steps=step, forwards=rule.forwards,
        train_seconds=round(train_time, 1), budget_minutes=minutes, device=device,
        params=model.n_params(), model=cfg.to_dict(), batch_size=batch_size, seed=seed,
        diverged=diverged, unigram_loss=data.unigram_loss, curve=curve, sample=sample,
    )
    if save:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"{rule_name}{'-' + tag if tag else ''}.json"
        out.write_text(json.dumps(result, indent=1))
        log(f"wrote {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override rule hyperparameters")
    a = ap.parse_args()
    if a.list or not a.rule:
        for name, cls in sorted(RULES.items()):
            print(f"{name:20s} [{cls.autograd:8s}] {cls.blurb}")
        print("\nautograd levels:", *(f"  {k:8s} {v}" for k, v in AUTOGRAD_LEVELS.items()), sep="\n")
        return
    run(a.rule, a.minutes, pick_device(a.device), parse_sets(a.set), GPTConfig(),
        batch_size=a.batch_size, seed=a.seed, tag=a.tag, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
