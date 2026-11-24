"""WIRE TOGETHER: "neurons that fire together wire together", taken literally.

No noise, no gradients, no learning rate, no optimisation of any kind. The
rule just counts which character follows which (a Hebbian co-occurrence
count between an input neuron and an output neuron) as batches stream past,
then writes those counts straight into the GPT's weights in closed form:

  * tok_emb[c]   = 30 x one-hot of c     -> each char gets its own neuron
  * every block's output projections = 0     -> blocks pass the stream through untouched
  * pos_emb      = 0
  * lm_head      = chosen so that lm_head(ln_f(onehot c)) = log P(next | c)

ln_f squashes the one-hots into 65 linearly independent vectors in 128-d, so
the readout that maps them to the log-probabilities is solved exactly with a
pseudo-inverse. The network becomes a bigram model wired by counting.

It is a floor, a joke, and a provocation: if the gradient-free learners can't
beat counting letter pairs, they haven't learned anything a census couldn't.
"""
import torch

from . import Rule, register


def compile_bigram(model, counts: torch.Tensor, smoothing: float, scale: float = 30.0):
    """Wire `model` so its forward pass computes the smoothed bigram model in `counts`."""
    V, C = model.cfg.vocab_size, model.cfg.n_embd
    assert C >= V, "needs n_embd >= vocab_size to give every char its own neuron"
    dev = model.lm_head.weight.device
    with torch.no_grad():
        model.tok_emb.weight.zero_()
        # big one-hots: ln_f erases the scale, so the bigram is identical, but anything the blocks
        # later leak into the stream gets amplified ~scale x less by ln_f (makes it a sturdy starting point)
        model.tok_emb.weight[:, :V] = scale * torch.eye(V, device=dev)
        model.pos_emb.weight.zero_()
        for blk in model.blocks:
            for lin in (blk.attn.proj, blk.mlp.proj):
                lin.weight.zero_()
                lin.bias.zero_()
        model.ln_f.weight.fill_(1.0)
        model.ln_f.bias.zero_()
        U = model.ln_f(model.tok_emb.weight)  # (V, C): what the readout sees for each char
        P = (counts + smoothing) / (counts + smoothing).sum(1, keepdim=True)
        L = P.log()  # (V_cur, V_next)
        # want U @ W^T = L  ->  W^T = pinv(U) @ L
        W = (torch.linalg.pinv(U.cpu().double()) @ L.cpu().double()).T
        model.lm_head.weight.copy_(W.float().to(dev))


@register
class WireTogether(Rule):
    name = "wire_together"
    autograd = "none"
    blurb = "Pure Hebbian counting: tally which char follows which, write the counts into the weights in closed form. Zero optimisation."
    paper = "Hebb 1949; it is a bigram model (Shannon 1948) wearing a transformer costume"
    defaults = dict(smoothing=0.1, recompile_every=50)
    tune = "smoothing"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        V = model.cfg.vocab_size
        self.counts = torch.zeros(V, V, device=model.lm_head.weight.device)
        self.k = 0

    @torch.no_grad()
    def step(self, x, y):
        V = self.counts.size(0)
        # fire together: (current char, next char) co-occurrences in this batch
        self.counts += torch.bincount((x * V + y).reshape(-1), minlength=V * V).view(V, V).float()
        if self.k % self.hp["recompile_every"] == 0:
            compile_bigram(self.model, self.counts, self.hp["smoothing"])
        self.k += 1
        self.forwards += 1  # it reads the batch once; it never even runs the network
        logits, loss = self.model(x, y)
        return loss.item()
