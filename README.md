# beyond-backprop 🧠⚡

**A playground for learning rules that aren't backprop, tested on a tiny GPT.**

Backprop works absurdly well, and it's also kind of weird: a global error signal, a
perfectly symmetric backward pass, every activation stored until the update. Brains
don't seem to do any of that. So what else can train a language model?

This repo is a small, honest arena for finding out. Every idea, sane or unhinged,
trains the **same 0.8M-param character-level GPT** on **TinyShakespeare** for the
**same wall-clock budget**, and lands on one leaderboard. It runs on a laptop
(Apple-silicon GPU, CUDA, or CPU) in minutes.

Most of these will lose to backprop. That's the fun part. The interesting question
is *by how much*, and *why*.

## Leaderboard

<!-- LEADERBOARD:START -->
*(results pending: run `python -m bbp.tournament`)*
<!-- LEADERBOARD:END -->

**How to read it**
- **val loss**: cross-entropy on held-out Shakespeare, nats per character. Lower is better.
  4.17 = uniform guessing over 65 chars, ~3.3 = knows letter frequencies only, ~1.5 = writes pseudo-Shakespeare.
- **autograd**: how much each rule leans on reverse-mode autodiff.
  🚫 `none` = no backward pass anywhere (**enforced**: the arena makes `.backward()` and `torch.autograd.grad` raise) ·
  🧱 `local` = backprop only *inside* a layer/block, never end to end ·
  🪞 `modified` = end-to-end backward pass, but not the true gradient ·
  ✅ `full` = backprop.
- **beats random reservoir?**: `reservoir` freezes the random network and trains only the
  final readout. If a rule can't beat that, its deep layers aren't really learning.
- **fwd passes**: total full-model forward passes spent. Wall-clock is the budget, but this
  shows how compute-hungry each idea is (ES burns 64 per step, backprop 1).
- **tuned**: each rule's one most important knob got the same quick sweep (×⅓, ×1, ×3,
  extended if the best was on the edge) before its final run. Nobody loses to a bad guess.

## The contenders

| rule | autograd | the idea | from |
|------|----------|----------|------|
| `backprop` | ✅ full | The baseline. AdamW. | Rumelhart, Hinton & Williams 1986 |
| `feedback_alignment` | 🪞 modified | The backward pass sends errors through **fixed random matrices** instead of Wᵀ. Somehow the forward weights learn to "align" with them. | Lillicrap et al. 2016 |
| `sign_symmetry` | 🪞 modified | Backward pass through `sign(W)`: the feedback knows the signs of the weights but not their size. | Liao, Leibo & Poggio 2016 |
| `local_losses` | 🧱 local | Every block gets its own tiny next-char readout and learns greedily. Gradients never cross a block boundary. | Belilovsky et al. 2019; Nøkland & Eidnes 2019 |
| `forward_forward` | 🧱 local | Hinton's FF for a causal LM: each block learns high "goodness" on real text, low on text with swapped chars. The readout is a probe on top. | Hinton 2022 |
| `reservoir` | 🧱 local | Random frozen transformer, only the readout trains. Sanity floor. | Jaeger 2001 |
| `forward_grad` | 🚫 none | One forward-mode JVP along a random direction v gives (∇L·v); use (∇L·v)v as the gradient. Unbiased, hugely noisy. | Baydin et al. 2022 |
| `mezo` | 🚫 none | SPSA: nudge all weights by ±εz, compare two losses, step along z. No autograd, no activation memory. | Spall 1992; Malladi et al. 2023 |
| `es` | 🚫 none | Evolution strategies: 64 noisy clones evaluated in parallel with `vmap`, ranked, weights drift toward the winners. | Salimans et al. 2017 |
| `hill_climb` | 🚫 none | Jiggle every weight at random; keep it if the loss dropped. The dumbest possible learner, as a floor. | Rechenberg 1973 |

## Quickstart

```bash
git clone https://github.com/ozkilim/beyond-backprop && cd beyond-backprop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m bbp.arena --list                          # what's in the arena
python -m bbp.arena --rule backprop --minutes 1     # train one rule, prints a Shakespeare sample
python -m bbp.arena --rule mezo --set lr=3e-4 eps=1e-3
python -m bbp.tournament                            # the full leaderboard (sweeps + 5-min finals, ~2h)
python -m bbp.leaderboard                           # rebuild the table + plot from results/
pytest                                              # every rule takes a few steps on a tiny CPU model
```

TinyShakespeare (1.1 MB) downloads itself on first run.

## Add your own crazy idea

One file, one class. Copy [`bbp/rules/_template.py`](bbp/rules/_template.py) to
`bbp/rules/your_idea.py`:

```python
from . import Rule, register

@register
class MyIdea(Rule):
    name = "my_idea"
    autograd = "none"      # be honest; "none" is enforced
    blurb = "Weights learn by vibes."
    defaults = dict(lr=1e-3)

    @torch.no_grad()
    def step(self, x, y):              # x, y: (B, T) char ids and next-char targets
        _, loss = self.model(x, y)
        self.forwards += 1
        ...                            # change self.model's weights however you like
        return loss.item()
```

It's picked up automatically. You can reach inside the model (`model.embed`,
`model.blocks[i]`, `model.head`) and bolt on your own modules
(see [`local_losses.py`](bbp/rules/local_losses.py)), as long as the model's normal
`forward()` is what gets evaluated. Then `python -m bbp.tournament --rules my_idea` and
open a PR with your result JSON.

### Wishlist: ideas nobody has put in the arena yet

- **Predictive coding**: every layer predicts the one below; learning = minimising local prediction errors (Rao & Ballard 1999; Whittington & Bogacz 2017)
- **Target propagation / difference target prop**: learned inverses send *targets*, not gradients, down the network (Lee et al. 2015)
- **Equilibrium propagation**: nudge the output, let the network settle, learn from the difference between the two equilibria (Scellier & Bengio 2017)
- **Direct feedback alignment**: random projections of the output error go straight to every layer (Nøkland 2016)
- **PEPITA**: a second forward pass with the error added to the *input* (Dellaferrera & Kreiman 2022)
- **Synthetic gradients**: each layer learns to *predict* its own gradient (Jaderberg et al. 2017)
- **NoProp**: each block learns to denoise a label embedding, diffusion style, with no forward or backward propagation through the stack (Li, Teh & Pascanu 2025)
- **Hebbian + neuromodulator**: "fire together, wire together", gated by a global reward/surprise signal
- **HSIC bottleneck**: layers maximise dependence on labels and minimise it on inputs (Ma et al. 2019)
- **Low-rank / sparse zeroth-order**: only perturb a random subspace or a few weights each step
- **Genetic crossover**: breed populations of tiny GPTs. Does recombination help?
- **LLM as optimizer**: show a big model the weights' stats and the loss and let it *propose* the update 🙃
- ...whatever you're thinking right now that sounds stupid. Those are the good ones.

## Fine print

- One seed, one 5-minute budget, one tiny model: this is a playground, not a paper.
  Rankings between close rules can flip with seeds or budgets; huge gaps won't.
- Wall-clock favours rules that map well to the GPU. `fwd passes` is there so you can
  argue about compute instead.
- Short budgets favour fast starters. Some of these ideas (FA, local losses) are known to
  close the gap or fall further behind with more training and depth.

MIT licensed. PRs with new rules, better hyperparameters, or longer runs are very welcome.
