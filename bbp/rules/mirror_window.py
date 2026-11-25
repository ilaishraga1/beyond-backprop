"""MIRROR WINDOW: grandmother_cells' wiring, but the neurons learn what to look for.

Attention is hand-wired to lay the last few characters side by side in the stream. All the
MLP neurons of all blocks together form ONE hidden layer of 2,048 units, and write straight
into 65 logit slots:

  * OUTPUT weights   delta rule:  (output error) x (how much did I fire)
  * INPUT weights    the output error arrives through a separate matrix B, times the neuron's
                     own slope, times its input
  * B itself         starts random and receives the same product as the output weights
                     (Kolen & Pollack 1994), so it lines up without any weight being copied

No autograd is called, but this is a two-layer chain rule with a learned stand-in for the
backward weights, so the label is 'modified', like feedback_alignment. Neurons start out
caring more about recent characters (input scale shrinks by `fade` per character back).
What is given, not learned: attention.

`feedback` ablations: "fixed" (B frozen = feedback alignment), "exact" (B = the output weights:
plain backprop on this little net, diagnostic), "frozen" (input weights never learn).
"""
import math

import torch
import torch.nn.functional as F

from . import Rule, register
from .grandmother_cells import CellModel


def _gelu_grad(a):
    return 0.5 * (1.0 + torch.erf(a / math.sqrt(2.0))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)


@register
class MirrorWindow(Rule):
    name = "mirror_window"
    autograd = "modified"
    blurb = "Hand-wired copy heads show 2,048 neurons the last few chars. Output weights: delta rule. Input weights: error through a separate matrix that receives the same Hebbian updates as its forward twin (no weight copying)."
    paper = "Kolen & Pollack 1994; Akrout et al. 2019; Bengio et al. 2003 (fixed-window neural LM)"
    defaults = dict(lr=1e-3, feedback="mirror", extra=1, gain=0.3, fade=0.75, half=8000)
    tune = "lr"

    @torch.no_grad()
    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.cm = cm = CellModel(model, extra=int(self.hp["extra"]))
        self.h = {}
        self.mask = torch.zeros(cm.C, device=cm.dev)
        self.mask[:cm.b * cm.N] = 1.0  # neurons see char codes only
        g = torch.Generator().manual_seed(int(torch.randint(2 ** 31, (1,)).item()))
        fade = torch.tensor([self.hp["fade"] ** (i // cm.b) if i < cm.b * cm.N else 0.0 for i in range(cm.C)])
        self.mlps = [blk.mlp for blk in model.blocks]
        self.B = []
        for i, m in enumerate(self.mlps):
            m.fc.weight.copy_((torch.randn(m.fc.weight.shape, generator=g) * self.hp["gain"] * fade).to(cm.dev) * cm.sd)
            m.fc.bias.zero_()
            m.proj.weight.zero_()
            self.B.append((torch.randn(cm.V, m.fc.out_features, generator=g) * 0.02 / cm.U).to(cm.dev))
            m.fc.register_forward_hook(lambda mod, inp, out, i=i: self.h.__setitem__(i, (inp[0], out)))
        # the stream is divided by sd before fc and multiplied by U after proj; Adam steps are scaled to match
        fc = [m.fc.weight for m in self.mlps]
        bias = [m.fc.bias for m in self.mlps]
        out = [m.proj.weight for m in self.mlps] + [self.mlps[-1].proj.bias]
        groups = [(out, self.hp["lr"] / cm.U), (self.B, self.hp["lr"] / cm.U)]
        if self.hp["feedback"] != "frozen":
            groups += [(fc, self.hp["lr"] * cm.sd), (bias, self.hp["lr"])]
        self.opt = torch.optim.Adam([dict(params=p, lr=lr, base=lr) for p, lr in groups], lr=self.hp["lr"])
        self.k = 0

    @torch.no_grad()
    def step(self, x, y):
        cm, fb = self.cm, self.hp["feedback"]
        for grp in self.opt.param_groups:
            grp["lr"] = grp["base"] * 0.5 ** (self.k / self.hp["half"])
        self.k += 1
        self.forwards += 1
        logits, loss = self.model(x, y)
        err = ((logits.softmax(-1) - F.one_hot(y, cm.V).float()) / y.numel()).reshape(-1, cm.V)
        for i, m in enumerate(self.mlps):
            u, a = self.h[i]
            a = a.reshape(-1, cm.n_fc)
            h = F.gelu(a)
            hebb = err.T @ h  # (V, n_fc): error x activity, seen by both twins
            m.proj.weight.grad = torch.zeros_like(m.proj.weight)
            m.proj.weight.grad[cm.l0:cm.l0 + cm.V] = hebb
            if fb != "frozen":
                B = self.B[i] if fb != "exact" else m.proj.weight[cm.l0:cm.l0 + cm.V]
                back = (err @ B) * _gelu_grad(a)
                m.fc.weight.grad = (back.T @ u.reshape(-1, cm.C)) * self.mask
                m.fc.bias.grad = back.sum(0)
            if fb == "mirror":
                self.B[i].grad = hebb
        gb = torch.zeros_like(self.mlps[-1].proj.bias)
        gb[cm.l0:cm.l0 + cm.V] = err.sum(0)
        self.mlps[-1].proj.bias.grad = gb
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        """Cosine between B and the output weights it stands in for."""
        cm = self.cm
        O = torch.cat([m.proj.weight[cm.l0:cm.l0 + cm.V] for m in self.mlps], 1)
        return torch.cosine_similarity(O.flatten(), torch.cat(self.B, 1).flatten(), 0).item()
