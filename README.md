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
| # | rule | autograd | test loss ↓ | vs backprop | beats random reservoir? | best ckpt at | steps | fwd passes | tuned |
|---|------|----------|-----------:|------------:|:--:|------:|------:|-----------:|-------|
| 1 | `backprop` | ✅ full | 1.629 | +0.000 | yes | 1.0 min | 23,871 | 23,871 | lr=0.00067 |
| 2 | `sign_symmetry` | 🪞 modified | 1.675 | +0.047 | yes | 2.0 min | 20,864 | 20,864 | lr=0.001 |
| 3 | `local_losses` | 🧱 local | 1.743 | +0.114 | yes | 2.7 min | 18,457 | 18,457 | lr=0.002 |
| 4 | `feedback_alignment` | 🪞 modified | 1.911 | +0.282 | yes | 4.5 min | 21,233 | 21,233 | lr=0.001 |
| 5 | `forward_forward` | 🧱 local | 2.698 | +1.069 | yes | 4.3 min | 10,107 | 20,214 | lr=0.009 |
| 6 | `reservoir` | 🧱 local | 2.716 | +1.088 | — | 3.8 min | 52,867 | 52,867 | lr=0.003 |
| 7 | `es` | 🚫 none | 2.788 | +1.160 | no | 5.0 min | 895 | 57,280 | lr=0.003 |
| 8 | `mezo` | 🚫 none | 2.818 | +1.190 | no | 5.0 min | 25,747 | 51,494 | lr=0.0009 |
| 9 | `hill_climb` | 🚫 none | 3.153 | +1.524 | no | 5.0 min | 24,757 | 49,514 | sigma=0.01 |
| 10 | `forward_grad` | 🚫 none | 3.167 | +1.539 | no | 5.0 min | 9,280 | 9,280 | lr=0.0018 |
| | *unigram (letter frequencies only)* | | 3.309 | | | | | | |
| | *untrained / uniform guess* | | 4.216 | | | | | | |

<sub>0.83M-param char GPT (4 layers, d=128, ctx 128), batch 32, TinyShakespeare, 5 min of training time per rule on `mps`, loss in nats/char. Early stopping: each rule's best checkpoint by val loss is scored on a held-out test split. Each rule's `tuned` knob got a 3-point sweep first.</sub>

![loss curves](results/curves.png)

<details><summary>What each rule's model writes after its run (prompt <code>ROMEO:</code>)</summary>

**backprop** (test 1.629)

```
ROMEO:
Thou must have sell me of your holy daughter,
Because to give thee that the house is son?
To him stands, your way, and come already, no; my
father we will not carriied to the head and her fruit,
I hold yet for his files foes, the maring of mercy,
Their still to their seating of the queen.

ME
```

**sign_symmetry** (test 1.675)

```
ROMEO:
Thou mayst have not made thee hanging.
I toe there courty he now here here would
The sense of these attors or way were comes;
And true nearnest of well stone and winders,
By the contrarch from the forth to fair it?

MERCUTIO:
Has their come?

POLIXENES:
They speed and grown be them.

EDWARD:

```

**local_losses** (test 1.743)

```
ROMEO:
Thou must know'st the decay?
That ways I to thee, I was you for me
To have think the sensuader by that of the way,
For commony thee, nor my father we would for me
I seek that the Lancaster,
I shall be my pray thee--
Whom on not say that of me for my in
This medicitizens. Boy! Fare not to the 
```

**feedback_alignment** (test 1.911)

```
ROMEO:
Tutry his kiss selves ouch of hang,
Nor gelse than of to giventreather strught she ever igns,
For that of to a to be ancesenay thour nee mothereing penty one.

So king?

LENRLEY:
I not sweear strue:
I knew farew dam, you of not say, if come defence,
And the me here, my pown, to the at to the 
```

**forward_forward** (test 2.698)

```
ROMEO:
ty remhis kess selteelouce neaman yodd Ihelsteterelo urey h t t beere heth tot s orel s s de hil
Wed tso y or i d we alcos idarthoer nd- mhaf ttoor s wustonodrimf Iiserdott,
Whesc b ts uwed

RSOCur wanofeprf pirir-O:
HATooennhe sarfeint o q d fl o I inou ind t h t d m.
Hoinoro y toresthase st
```

**reservoir** (test 2.716)

```
ROMEO:
TENGLOhis kARGLs:
WIULoHER heamanUCaid INtlstherd houre y he ntheeare hath hou seerel s hanorre tsthathory we w akeraricothiay thofa nd- myt thyoo we wootour aim wIiouovoto d te c an s u.
B
Tu th.
So I fe EN ithi

Oilyompou noussay fiot ou ld fo o I igl bimy t u ou,
Bs toinout y tonouthasof t
```

**es** (test 2.788)

```
ROMEO:
cu remhis khes selteeloH t meaman yodd Igels e gMeco ue y he nte here he?

Sot s chel s me onre t t atho y we w apee ancos id
J


:
And d han thoo we wost te acif Iinerethond te c an s uz


IRSnr.SSo I

T3rf porir mily moon n t say fins od ld fe ome inl zMThee t thon s
I:


FS y tonoon gor ha
```

**mezo** (test 2.818)

```
ROMEO:
EW remhit hhes seltemlouca meamay yayd Iotlste gMeloyae y he ntheeere heth thin be elis ig sere t thathory we w ky e ancos yay thoer nd t faf theo we w y t f ar'f Iioerdot ad t wc an s uw the s h. fo Iofend, pur and lr mooennd ssay tint od ld fl ome inl zind t s t sn s t inobt y ton t t sof t
```

**hill_climb** (test 3.153)

```
ROMEO:
ts re hOt khawhs:lte toHct me manlrodd Ig
ls e gMe o ue y h thty
here hath thoos thelod so the mose atso y wr w k we aRces iaretho:r ndn mot t tod we w yt donrim d iodovot , thes  an s uw t
ur nr.S w Inyepr, ithi

welyAmoonnn,rssay ed t odeld fl ome inle innleps t dnm.
t in,dS y tononEng t  t
```

**forward_grad** (test 3.167)

```
ROMEO:
t
 re h

d t wLselteeloH u neaheorrocd Ig
lst tgM ccfueoy n fnt bhhre h th thib me el sBeg sehe t t athof ler w k w

ahees uare fotr nene th theoor s w yt te ai'f Iioeotot ad t s  an s uz f

e n..Se aIofeorf pe ne mily mooe n t say eiI codele fleomeeinor imkee s t dn se.oinobe y th tot g r s 
```

</details>
<!-- LEADERBOARD:END -->

**What the first run says** (one seed, 5 minutes, so hold it loosely)
- **Backprop still wins**, and it gets there fast: it peaks after 1 minute, then starts memorising.
- **Sign symmetry is within 0.05 nats.** A backward pass that only knows the *signs* of
  the weights, not their size, is nearly as good as the real gradient here.
- **Local per-block losses come third (+0.11)** with no gradient ever crossing a block
  boundary, and they overfit far more slowly than backprop.
- **Feedback alignment (+0.28) is still improving at minute 5.** Random feedback works,
  just slower, because the forward weights first have to "align" with it.
- **Forward-Forward only barely beats a frozen random network** (2.698 vs 2.716): its blocks
  learn something, but almost none of it helps predict the next character.
- **Every rule with no backward pass at all loses to the frozen random network.** ES and
  MeZO are still improving at minute 5, though; this is the bracket where longer budgets
  and better ideas have the most room. Go win it.

**How to read it**
- **test loss**: cross-entropy on held-out Shakespeare, nats per character. Lower is better.
  4.17 = uniform guessing over 65 chars, ~3.3 = knows letter frequencies only, ~1.5 = writes pseudo-Shakespeare.
- **best ckpt at / early stopping**: TinyShakespeare is small. Backprop rips through ~90 epochs
  in 5 minutes and starts memorising the training text after about a minute. Scoring the
  *last* weights would reward rules for being bad at overfitting, so the data is split
  90/5/5 train/val/test: every rule keeps its best checkpoint by val loss, and that
  checkpoint is scored on a test split it has never seen. "best ckpt at" shows when each rule peaked.
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
- Test loss runs ~0.1 nats above val for everyone: test is the last 5% of the corpus (different
  plays), val the 5% before it. It's the same shift for every rule, so rankings are comparable.
- Wall-clock favours rules that map well to the GPU. `fwd passes` is there so you can
  argue about compute instead.
- An early version of this arena scored the final weights, and local losses "beat" backprop
  by 0.27 nats. It was pure overfitting: backprop had peaked at minute 1.3 and spent the
  rest memorising. Hence early stopping. If your rule suddenly beats backprop, check the
  curve before you celebrate.
- Short budgets favour fast starters. Some of these ideas (FA, local losses) are known to
  close the gap or fall further behind with more training and depth.

MIT licensed. PRs with new rules, better hyperparameters, or longer runs are very welcome.
