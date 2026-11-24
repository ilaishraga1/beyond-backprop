"""LAB RATS: every token is its own experiment.

Weight-perturbation rules (mezo, es, hill_climb) jiggle all 830k weights at
once and get ONE number back per forward pass: did the loss go up or down?
Hopeless signal-to-noise.

Here we jiggle *neurons* instead of weights, and we do it independently at
every token. A batch is 32 x 128 = 4,096 tokens, so one pair of forward passes
runs 4,096 tiny experiments in parallel:

  1. forward with +noise on the output of every Linear layer (and the embedding),
     fresh noise for every token and every neuron
  2. forward again with the *same* noise flipped to -noise
  3. per token, the loss difference says whether that token's jolt helped:
         dL/dy  ~=  (L+ - L-) / (2 sigma^2) * noise
  4. each layer turns that into a weight update the Hebbian way, as an outer
     product with its own input:  dW = sum_tokens  dL/dy (x) input

No backward pass, no autograd, no weight transport. Every update is local to
its layer; the only global thing is the per-token loss, broadcast like a
dopamine signal.

The cheat it makes: a token's loss difference is credited only to the noise at
*that* token, ignoring how it leaked into later tokens through attention.
Biased, but orders of magnitude less noisy.

Lineage: node perturbation (Williams 1992; Fiete & Seung 2006; Dalm et al. 2023).
The per-token-credit twist is the crazy bit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Rule, register


@register
class LabRats(Rule):
    name = "lab_rats"
    autograd = "none"
    blurb = "Jolt every neuron at every token, 4,096 parallel experiments per batch; per-token loss change -> Hebbian outer-product updates."
    paper = "node perturbation (Williams 1992; Fiete & Seung 2006) + per-token credit in a causal LM (the twist tried here)"
    defaults = dict(lr=1e-3, sigma=0.03)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.sites = self.lins + [model.tok_emb]  # where noise is injected
        self.mode = 0  # 0 = clean (evaluation), +1 / -1 = antithetic noisy passes
        self.noise, self.inputs = {}, {}
        for m in self.sites:
            m.register_forward_hook(self._hook)
        params = [p for m in self.lins for p in m.parameters()]
        params += [model.tok_emb.weight, model.pos_emb.weight]
        self.opt = torch.optim.Adam(params, lr=self.hp["lr"])

    def _hook(self, mod, inp, out):
        if self.mode == 0:
            return None
        if self.mode == 1:
            self.noise[mod] = torch.randn_like(out) * self.hp["sigma"]
            self.inputs[mod] = inp[0]
        else:  # average the + and - inputs: ~ the clean input, to first order
            self.inputs[mod] = 0.5 * (self.inputs[mod] + inp[0]) if inp[0].is_floating_point() else inp[0]
        return out + self.mode * self.noise[mod]

    def _token_losses(self, x, y):
        logits, _ = self.model(x)
        return F.cross_entropy(logits.transpose(1, 2), y, reduction="none")  # (B, T)

    @torch.no_grad()
    def step(self, x, y):
        sigma = self.hp["sigma"]
        self.mode = 1
        lp = self._token_losses(x, y)
        self.mode = -1
        lm = self._token_losses(x, y)
        self.mode = 0
        coef = (lp - lm) / (2 * sigma**2)  # (B, T): how much each token's jolt hurt
        N = x.numel()

        for m in self.lins:
            g = coef.unsqueeze(-1) * self.noise[m]  # dL/d(output), per token
            xin = self.inputs[m]
            m.weight.grad = torch.einsum("bto,bti->oi", g, xin) / N  # Hebbian outer product
            if m.bias is not None:
                m.bias.grad = g.sum((0, 1)) / N
        # embedding: the jolt on tok_emb's output is a jolt on (tok_emb + pos_emb)
        emb = self.model.tok_emb
        g = (coef.unsqueeze(-1) * self.noise[emb]).reshape(-1, emb.weight.size(1))
        emb.weight.grad = torch.zeros_like(emb.weight).index_add_(0, x.reshape(-1), g) / N
        pos = self.model.pos_emb.weight
        pos.grad = torch.zeros_like(pos)
        pos.grad[: x.size(1)] = g.view(*x.shape, -1).sum(0) / N

        self.opt.step()
        self.noise.clear(); self.inputs.clear()
        self.forwards += 2
        return 0.5 * (lp.mean().item() + lm.mean().item())
