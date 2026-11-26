"""A screening split that behaves like test. The arena's val (the 5% after train) is mostly
the same play the training text ends in; test is mostly a new one (The Tempest). So tricks
picked on val often fail on test. For screening ideas, hold out a whole play from the middle
of train instead (Richard II, chars 28.4%-39.6%: 24 speakers, only 4 seen elsewhere in train)
and use it to pick checkpoints. The arena's real val and test are never touched.

    import bbp.proxy; bbp.proxy.use()   # before calling bbp.arena.run; best_val is then the proxy loss
"""
import torch

import bbp.arena as arena
from bbp.data import CharData

LO, HI = 0.284, 0.396


class ProxyData(CharData):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        full = torch.cat([self.splits["train"], self.splits["val"], self.splits["test"]])
        n, tr_end = len(full), len(self.splits["train"])
        lo, hi = int(n * LO), int(n * HI)
        self.splits = {"train": torch.cat([full[:lo], full[hi:tr_end]]), "val": full[lo:hi], "test": full[lo:hi]}
        self.train = self.splits["train"]


def use():
    arena.CharData = ProxyData
