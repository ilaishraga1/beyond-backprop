"""NATURAL SELECTION: neurons compete, starve and breed.

Starts with 65 neurons, one per character, knowing nothing. Then life begins:

  * LEARN   every cell's output weights follow the delta rule: (output error) x (did I fire?)
            One layer, no chain rule through any hidden unit (still a gradient, so: local).
  * DIE     each generation the cells that move the logits least are killed
  * BREED   a newborn appears where the model was most surprised: pick a token in proportion
            to its loss, pick a cell that fired there, copy it, and mutate what it looks for:
            demand one more char that was really there, or forget one (a wildcard)

Which strings the hidden neurons detect is never touched by an error gradient, only by
birth and death. Note the mutations allow gaps ("t?e"), which counting suffixes cannot hire.

Ablations: evolve=False (delta rule only), surprise=False (births at random tokens), fit="random" (blind deaths),
start="census" (begin from a fully hired grandmother_cells network instead).
"""
import torch
import torch.nn.functional as F

from . import Rule, register
from .grandmother_cells import GrandmotherCells


@register
class NaturalSelection(GrandmotherCells):
    name = "natural_selection"
    autograd = "local"  # the delta rule IS the one-layer readout gradient (same standard as reservoir)
    blurb = "Neurons that live: start with 65 single-char cells; output weights follow the delta rule, the least useful neurons die each generation, and mutants are born where the model is most surprised."
    paper = "Edelman 1987 (neural Darwinism); Fahlman & Lebiere 1990 (growing networks); Widrow & Hoff 1960 (delta rule)"
    defaults = dict(GrandmotherCells.defaults, lr=3e-3, gen_every=50, cull=0.05, evolve=True, surprise=True, fit="use", start="chars", count_steps=1000)
    tune = "lr"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.h = {}
        for i, blk in enumerate(model.blocks):
            blk.mlp.fc.register_forward_hook(lambda m, a, o, i=i: self.h.__setitem__(i, o))
        self.projs = [b.mlp.proj for b in model.blocks]
        self.scale = self.cm.A * self.cm.U
        self.opt = torch.optim.Adam([p.weight for p in self.projs] + [self.projs[-1].bias], lr=self.hp["lr"] / self.scale)
        self.rate = torch.zeros(self.cm.n_cells, device=self.cm.dev)
        self.born = 0

    def _begin(self):
        if self.hp["start"] == "chars":
            V = self.cm.V
            self.cells = [[(0, c)] for c in range(V)]
            self.cm.keys(self.cells, torch.zeros(V, V), torch.zeros(V))
        self.cells = [tuple(sorted(c)) for c in self.cells] + [None] * (self.cm.n_cells - len(self.cells))
        self.alive = set(c for c in self.cells if c)

    def _generation(self, x, logits, y):
        cm = self.cm
        w = torch.cat([p.weight[cm.l0:cm.l0 + cm.V].T for p in self.projs])
        fit = (self.rate * w.pow(2).sum(1)).cpu()  # how much this cell moves the logits
        if self.hp["fit"] == "random":  # control: death is blind
            fit = torch.rand(cm.n_cells)
        fit[[i for i, c in enumerate(self.cells) if c is None]] = -1.0
        n = max(1, int(self.hp["cull"] * cm.n_cells))
        doomed = fit.argsort()[:n].tolist()
        loss = F.cross_entropy(logits.reshape(-1, cm.V), y.reshape(-1), reduction="none").view_as(y).clone()
        if not self.hp["surprise"]:
            loss = torch.ones_like(loss)
        loss[:, :cm.N] = 0  # no full history yet
        spots = torch.multinomial(loss.reshape(-1).cpu(), n, replacement=True)
        T = y.size(1)
        fired = torch.cat([self.h[i].reshape(-1, cm.n_fc)[spots.to(cm.dev)] for i in range(len(self.projs))], 1) > 0
        fired, xs = fired.cpu(), x.cpu()
        new, dead = [], set(doomed)
        for slot, s, row in zip(doomed, spots.tolist(), fired):
            parents = [i for i in row.nonzero().flatten().tolist() if self.cells[i] and i not in dead]
            if not parents:
                continue
            p = self.cells[parents[torch.randint(len(parents), (1,)).item()]]
            b, t = divmod(s, T)
            free = [k for k in range(cm.N) if k not in [q for q, _ in p]]
            if len(p) > 1 and (not free or torch.rand(1).item() < 0.25):
                i = torch.randint(len(p), (1,)).item()
                child = p[:i] + p[i + 1:]
            else:
                # likelier to want a nearby char than a far one
                k = free[int(torch.multinomial(torch.tensor([0.5 ** f for f in free]), 1))]
                child = tuple(sorted(p + ((k, int(xs[b, t - k])),)))
            if child not in self.alive:
                self.alive.add(child)
                new.append((slot, child))
        for slot, child in new:
            if self.cells[slot]:
                self.alive.discard(self.cells[slot])
            self.cells[slot] = child
            self.born += 1
        if not new:
            return
        fw, fb = torch.zeros(len(new), cm.C), torch.zeros(len(new))
        for i, (_, child) in enumerate(new):
            for back, ch in child:
                fw[i, cm.b * back:cm.b * (back + 1)] = cm.A / 2 * cm.sd * cm.code[ch]
            fb[i] = cm.A - cm.A / 2 * cm.b * len(child)
        fw, fb = fw.to(cm.dev), fb.to(cm.dev)
        slots = torch.tensor([s for s, _ in new])
        for bi, blk in enumerate(self.model.blocks):
            m = slots // cm.n_fc == bi
            if not m.any():
                continue
            col, m = (slots[m] % cm.n_fc).to(cm.dev), m.to(cm.dev)
            blk.mlp.fc.weight[col], blk.mlp.fc.bias[col] = fw[m], fb[m]
            blk.mlp.proj.weight[:, col] = 0.0
            for key in ("exp_avg", "exp_avg_sq"):
                st = self.opt.state.get(blk.mlp.proj.weight)
                if st:
                    st[key][:, col] = 0.0
        self.rate[slots.to(cm.dev)] = 0.0

    @torch.no_grad()
    def step(self, x, y):
        counting = self.k < int(self.hp["count_steps"]) and self.hp["start"] == "census"
        if counting:
            return super().step(x, y)
        if not hasattr(self, "alive"):
            self._begin()
        self.k += 1
        self.forwards += 1
        cm = self.cm
        logits, loss = self.model(x, y)
        err = (logits.softmax(-1) - F.one_hot(y, cm.V).float()) / y.numel()
        flat = err.reshape(-1, cm.V)
        for i, p in enumerate(self.projs):
            h = F.gelu(self.h[i]).reshape(-1, cm.n_fc)
            g = torch.zeros_like(p.weight)
            g[cm.l0:cm.l0 + cm.V] = flat.T @ h
            p.weight.grad = g
            sl = slice(i * cm.n_fc, (i + 1) * cm.n_fc)
            self.rate[sl] = 0.99 * self.rate[sl] + 0.01 * (h > 0).float().mean(0)
        gb = torch.zeros_like(self.projs[-1].bias)
        gb[cm.l0:cm.l0 + cm.V] = flat.sum(0)
        self.projs[-1].bias.grad = gb
        self.opt.step()
        if self.hp["evolve"] and self.k % int(self.hp["gen_every"]) == 0:
            self._generation(x, logits, y)
        return loss.item()
