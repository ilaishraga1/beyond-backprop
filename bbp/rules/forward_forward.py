"""Hinton's Forward-Forward, adapted to a causal LM.

Each block is trained on its own to have high "goodness" (mean squared
activation) at positions of real text and low goodness at positions where a
character was swapped for a random one. Blocks see detached inputs, so
nothing propagates between them. The final readout (ln_f + lm_head) is then
trained as a local linear probe on the top block's detached features, so we
can measure next-char loss like everyone else.
"""
import torch
import torch.nn.functional as F

from . import Rule, register


@register
class ForwardForward(Rule):
    name = "forward_forward"
    autograd = "local"
    blurb = "Per-block goodness: high on real text, low on corrupted chars. Readout is a probe on top. No end-to-end signal."
    paper = "Hinton 2022, 'The Forward-Forward Algorithm'"
    defaults = dict(lr=1e-3, corrupt=0.15, threshold=2.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.hp["lr"])

    def _goodness(self, h, h_in):
        # goodness of what the block *added* to the residual stream
        return (h - h_in).pow(2).mean(-1)  # (B, T)

    def step(self, x, y):
        V, th = self.model.cfg.vocab_size, self.hp["threshold"]
        flip = torch.rand(x.shape, device=x.device) < self.hp["corrupt"]
        x_neg = torch.where(flip, torch.randint_like(x, V), x)
        # a position is "bad" if any char at or before it was corrupted (causal model)
        bad = flip.cumsum(1) > 0

        hp_, hn_ = self.model.embed(x), self.model.embed(x_neg)
        ff_loss = 0.0
        for i, blk in enumerate(self.model.blocks):
            inp_p = hp_ if i == 0 else hp_.detach()
            inp_n = hn_ if i == 0 else hn_.detach()
            hp_, hn_ = blk(inp_p), blk(inp_n)
            gp, gn = self._goodness(hp_, inp_p), self._goodness(hn_, inp_n)
            pos = F.softplus(th - gp).mean()               # real text: push goodness up
            neg = F.softplus(gn - th)[bad].mean() if bad.any() else gn.sum() * 0  # corrupted: push down
            ff_loss = ff_loss + pos + neg
        logits = self.model.head(hp_.detach())  # probe: trains only ln_f + lm_head
        probe = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        self.opt.zero_grad(set_to_none=True)
        (ff_loss + probe).backward()
        self.opt.step()
        self.forwards += 2
        return probe.item()
