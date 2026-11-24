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
    # solo: "off"   = jolt every layer at once (one pair of forward passes per step)
    #       "cycle" = jolt one group per step, round-robin (same cost, each group updated 1/G of steps)
    #       "each"  = one pair of passes per group, every group updated every step (G x the cost)
    # groups = [embedding + block 0, block 1, ..., block n-1, lm_head]
    # gamma: cross-token credit. A jolt at token t can only change the loss at tokens s >= t
    #        (causal attention), so credit it with the discounted "loss-change-to-go"
    #        sum_{s>=t} gamma^(s-t) * dL_s, exactly like returns-to-go in RL.
    #        gamma=0 -> only token t's own loss (the original); gamma=1 -> unbiased, noisiest.
    # gamma_on: "attn" = only the attention qkv layers get cross-token credit (the only place a
    #        token's jolt directly reaches other tokens); "all" = every jolted site does.
    # local: the lowest `local` blocks each get a little next-char readout of their own (a "local
    #        maze"). Their jolts are credited with the change in THAT readout's loss, one step
    #        away, instead of the final loss far above (+ `mix` x the global credit). The
    #        readouts learn by jolting too. Nothing ever runs backward.
    defaults = dict(lr=1e-3, sigma=0.03, solo="off", gamma=0.0, gamma_on="attn", local=0, mix=0.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
        lin_of = lambda mod: [m for m in mod.modules() if isinstance(m, nn.Linear)]
        self.groups = [[model.tok_emb] + lin_of(model.blocks[0])]
        self.groups += [lin_of(b) for b in model.blocks[1:]]
        self.groups += [[model.lm_head]]
        self.active = set()  # sites that get jolted in the current pass
        self.mode = 0  # 0 = clean (evaluation), +1 / -1 = antithetic noisy passes
        self.noise, self.inputs = {}, {}
        self.turn = 0
        self.attn_sites = {b.attn.qkv for b in model.blocks}
        # local readouts ("mazes") for the lowest blocks
        C, V = model.cfg.n_embd, model.cfg.vocab_size
        dev = model.lm_head.weight.device
        self.aux = nn.ModuleList([nn.Linear(C, V, bias=False) for _ in range(self.hp["local"])]).to(dev)
        for a in self.aux:
            nn.init.normal_(a.weight, std=0.02)
        self.block_of = {model.tok_emb: 0}
        for i, g in enumerate(self.groups[: len(model.blocks)]):
            self.block_of.update({m: i for m in g if m is not model.tok_emb})
        for i, a in enumerate(self.aux):
            self.groups[i].append(a)
            self.block_of[a] = i
        self.block_out = {}
        for i in range(len(self.aux)):
            model.blocks[i].register_forward_hook(self._grab(i))
        for m in self.lins + [model.tok_emb] + list(self.aux):
            m.register_forward_hook(self._hook)
        params = [p for m in self.lins + list(self.aux) for p in m.parameters()]
        params += [model.tok_emb.weight, model.pos_emb.weight]
        self.opt = torch.optim.Adam(params, lr=self.hp["lr"])

    def _hook(self, mod, inp, out):
        if self.mode == 0 or mod not in self.active:
            return None
        if self.mode == 1:
            self.noise[mod] = torch.randn_like(out) * self.hp["sigma"]
            self.inputs[mod] = inp[0]
        else:  # average the + and - inputs: ~ the clean input, to first order
            self.inputs[mod] = 0.5 * (self.inputs[mod] + inp[0]) if inp[0].is_floating_point() else inp[0]
        return out + self.mode * self.noise[mod]

    def _grab(self, i):
        def hook(mod, inp, out):
            if self.mode != 0:
                self.block_out[i] = out
        return hook

    def _token_losses(self, x, y):
        """Per-token global loss (B, T), plus each local readout's per-token loss."""
        logits, _ = self.model(x)
        ce = lambda lg: F.cross_entropy(lg.transpose(1, 2), y, reduction="none")
        aux = [ce(a(F.layer_norm(self.block_out[i], self.block_out[i].shape[-1:]))) for i, a in enumerate(self.aux)]
        return ce(logits), aux

    def _credit_matrix(self, T, device):
        """M[s, t] = gamma^(s-t) for s >= t, so (dL @ M)[t] = sum_{s>=t} gamma^(s-t) dL[s]."""
        if getattr(self, "_M", None) is None or self._M.size(0) != T:
            i = torch.arange(T, device=device)
            lag = (i[:, None] - i[None, :]).float()  # s - t
            self._M = torch.where(lag >= 0, self.hp["gamma"] ** lag.clamp(min=0), torch.zeros_like(lag))
        return self._M

    def _experiment(self, x, y, sites):
        """One antithetic pair of passes jolting `sites`; writes their .grad. Returns mean loss."""
        sigma = self.hp["sigma"]
        self.active = set(sites)
        self.mode = 1
        lp, ap = self._token_losses(x, y)
        self.mode = -1
        lm, am = self._token_losses(x, y)
        self.mode = 0
        coef = (lp - lm) / (2 * sigma**2)  # (B, T): how much each token's jolt hurt
        coef_local = [(a - b) / (2 * sigma**2) for a, b in zip(ap, am)]  # same, for each local maze
        coef_future = coef
        if self.hp["gamma"] > 0:
            coef_future = coef @ self._credit_matrix(x.size(1), coef.device)  # loss-change-to-go
        N = x.numel()
        for m in sites:
            c = coef_future if (self.hp["gamma_on"] == "all" or m in self.attn_sites) else coef
            i = self.block_of.get(m, -1)
            if 0 <= i < len(self.aux):  # lives in a block with its own maze
                c = coef_local[i] if m is self.aux[i] else coef_local[i] + self.hp["mix"] * c
            g = c.unsqueeze(-1) * self.noise[m]  # dL/d(output), per token
            if m is self.model.tok_emb:
                # a jolt on tok_emb's output is a jolt on (tok_emb + pos_emb)
                g = g.reshape(-1, m.weight.size(1))
                m.weight.grad = torch.zeros_like(m.weight).index_add_(0, x.reshape(-1), g) / N
                pos = self.model.pos_emb.weight
                pos.grad = torch.zeros_like(pos)
                pos.grad[: x.size(1)] = g.view(*x.shape, -1).sum(0) / N
                continue
            m.weight.grad = torch.einsum("bto,bti->oi", g, self.inputs[m]) / N  # Hebbian outer product
            if m.bias is not None:
                m.bias.grad = g.sum((0, 1)) / N
        self.noise.clear(); self.inputs.clear(); self.block_out.clear(); self.active = set()
        self.forwards += 2
        return 0.5 * (lp.mean().item() + lm.mean().item())

    @torch.no_grad()
    def step(self, x, y):
        self.opt.zero_grad(set_to_none=True)  # Adam skips params that got no estimate this step
        solo = self.hp["solo"]
        if solo == "off":
            loss = self._experiment(x, y, [m for g in self.groups for m in g])
        elif solo == "cycle":
            loss = self._experiment(x, y, self.groups[self.turn % len(self.groups)])
            self.turn += 1
        elif solo == "each":
            loss = sum(self._experiment(x, y, g) for g in self.groups) / len(self.groups)
        else:
            raise ValueError(f"solo must be off/cycle/each, got {solo!r}")
        self.opt.step()
        return loss


@register
class LabRatsSolo(LabRats):
    name = "lab_rats_solo"
    blurb = "lab_rats, but one block jolted per pass pair (5 pairs/step): far less crosstalk, 5x the forwards."
    defaults = {**LabRats.defaults, "solo": "each"}


@register
class LabRatsCycle(LabRats):
    name = "lab_rats_cycle"
    blurb = "lab_rats, but only one block jolted per step, round-robin: less crosstalk at the same cost, each block updated 1/5 of steps."
    defaults = {**LabRats.defaults, "solo": "cycle"}


@register
class LabRatsFuture(LabRats):
    name = "lab_rats_future"
    blurb = "lab_rats, but attention jolts are credited with the discounted loss change at all *later* tokens (returns-to-go, gamma=0.9)."
    defaults = {**LabRats.defaults, "gamma": 0.9, "gamma_on": "attn"}


@register
class LabRatsMaze(LabRats):
    name = "lab_rats_maze"
    blurb = "lab_rats, but block 0 gets its own little next-char readout and is credited mostly by that nearby loss (+30% global)."
    defaults = {**LabRats.defaults, "lr": 1 / 3 * 1e-3, "local": 1, "mix": 0.3}
