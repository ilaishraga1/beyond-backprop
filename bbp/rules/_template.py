"""TEMPLATE: copy me to bbp/rules/<your_idea>.py and go wild.

Files starting with "_" are not loaded, so this one never shows up in the arena.

Checklist
  1. pick a unique `name` (CLI + leaderboard key)
  2. be honest about `autograd`:
       "none"     - no .backward() / autograd.grad anywhere (the arena enforces this)
       "local"    - backprop allowed only inside a layer/block
       "modified" - end-to-end backward pass, but not the true gradient
       "full"     - you're just doing backprop with extra steps
  3. step(x, y) gets a batch (B, T) of char ids and next-char targets.
     Update self.model however you like, return the training loss as a float.
  4. add to self.forwards the number of full forward passes you spent.
  5. run it:  python -m bbp.arena --rule <name> --minutes 1
     then the full 5-minute run, then python -m bbp.leaderboard

You can reach into the model: model.embed(x), model.blocks[i](h), model.head(h),
model.named_parameters(), model.cfg. You can add your own modules too
(see local_losses.py), just keep the model's normal forward() as the thing
that gets evaluated.
"""
import torch

from . import Rule, register


# @register   <- uncomment in your copy
class MyCrazyIdea(Rule):
    name = "my_crazy_idea"
    autograd = "none"
    blurb = "One line: what's the idea?"
    paper = ""
    defaults = dict(lr=1e-3)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)

    @torch.no_grad()
    def step(self, x, y):
        _, loss = self.model(x, y)
        self.forwards += 1
        # ... change self.model's weights here ...
        return loss.item()
