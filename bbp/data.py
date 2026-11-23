"""TinyShakespeare, character level. ~1.1M chars, 65 symbols.

Downloaded once into data/ on first use.
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class CharData:
    def __init__(self, block_size: int, device: str, val_frac: float = 0.1):
        path = DATA_DIR / "tinyshakespeare.txt"
        if not path.exists():
            DATA_DIR.mkdir(exist_ok=True)
            print(f"downloading TinyShakespeare -> {path}")
            try:
                urllib.request.urlretrieve(URL, path)
            except Exception:  # e.g. python.org macOS builds without certs installed
                subprocess.run(["curl", "-fsSL", URL, "-o", str(path)], check=True)
        text = path.read_text()
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train, self.val = data[:n], data[n:]
        self.block_size = block_size
        self.device = device
        # unigram entropy of the training set: the "learned nothing but letter frequencies" floor
        counts = torch.bincount(self.train, minlength=self.vocab_size).float()
        p = counts / counts.sum()
        self.unigram_loss = float(-(p[p > 0] * p[p > 0].log()).sum())

    def batch(self, split: str, batch_size: int, generator: torch.Generator | None = None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([d[i : i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)
