"""INSTINCT RATS: nature, then nurture.

Born with instincts: for the first `count_steps` batches the network learns
nothing but letter-pair counts (see wire_together.py), which are then wired
into its weights. Then it grows up: lab_rats_maze takes over and keeps
learning on top by jolting neurons, with no backward pass ever.

Question it asks: can the gradient-free learner get *past* the bigram, or does
it only ever re-discover what counting gives for free?
"""
from . import Rule, register
from .lab_rats import LabRatsMaze
from .wire_together import WireTogether, compile_bigram


@register
class InstinctRats(Rule):
    name = "instinct_rats"
    autograd = "none"
    blurb = "Born wired as a bigram (Hebbian counts), then lab_rats_maze keeps learning on top. Nature, then nurture."
    paper = "wire_together + lab_rats_maze (both this repo)"
    defaults = dict(lr=1e-4, count_steps=100, smoothing=0.1, scale=10.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.counter = WireTogether(model, smoothing=self.hp["smoothing"], recompile_every=10**9)
        self.rats = LabRatsMaze(model, lr=self.hp["lr"])  # its hooks stay idle until it steps
        self.k = 0

    def step(self, x, y):
        self.k += 1
        if self.k <= self.hp["count_steps"]:
            loss = self.counter.step(x, y)
            self.forwards = self.counter.forwards
            if self.k == self.hp["count_steps"]:  # childhood over: wire the instincts in
                compile_bigram(self.model, self.counter.counts, self.hp["smoothing"], self.hp["scale"])
            return loss
        loss = self.rats.step(x, y)
        self.forwards = self.counter.forwards + self.rats.forwards
        return loss
