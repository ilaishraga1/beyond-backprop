"""Evolution strategies: a population of noisy copies of the network, all
evaluated in parallel (vmap), ranked, and the weights move toward the winners.
"""
import torch
from torch.func import functional_call, vmap
import torch.nn.functional as F

from . import Rule, register
from ._util import params_of, set_grads


@register
class EvolutionStrategies(Rule):
    name = "es"
    autograd = "none"
    blurb = "OpenAI-style ES: antithetic population evaluated with vmap, rank-shaped fitness, Adam on the estimate."
    paper = "Salimans et al. 2017, 'Evolution Strategies as a Scalable Alternative to RL'"
    defaults = dict(lr=1e-3, sigma=0.01, pairs=32, chunk=16)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.opt = torch.optim.Adam(model.parameters(), lr=self.hp["lr"])

    @torch.no_grad()
    def step(self, x, y):
        P, sigma = self.hp["pairs"], self.hp["sigma"]
        params = params_of(self.model)
        half = {k: torch.randn((P, *p.shape), device=p.device) for k, p in params.items()}
        eps = {k: torch.cat([e, -e]) for k, e in half.items()}  # antithetic
        pop = {k: params[k].unsqueeze(0) + sigma * eps[k] for k in params}

        def loss_of(p):
            logits, _ = functional_call(self.model, p, (x,))
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        losses = vmap(loss_of, chunk_size=self.hp["chunk"])(pop)  # (2P,)
        ranks = losses.argsort().argsort().float()
        fit = ranks / (len(ranks) - 1) - 0.5  # centered ranks in [-0.5, 0.5], high = bad
        grad = {k: torch.einsum("n,n...->...", fit, eps[k]) / (len(fit) * sigma) for k in params}
        set_grads(self.model, grad)
        self.opt.step()
        self.forwards += 2 * P
        return losses.mean().item()
