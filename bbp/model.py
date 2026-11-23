"""A deliberately tiny GPT (~0.8M params at the default size).

Written so learning rules can get at the pieces:
  model.embed(idx)  -> residual stream x  (B, T, C)
  model.blocks[i](x) -> x                 (each block is pre-LN, residual)
  model.head(x)     -> logits             (final LayerNorm + unembedding)
  model(idx, y)     -> (logits, loss)

Attention is written out by hand (no fused SDPA kernel) so forward-mode AD
(torch.func.jvp) and vmap work on every backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, idx):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        return self.tok_emb(idx) + self.pos_emb(pos)

    def head(self, x):
        return self.lm_head(self.ln_f(x))

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 0.8, generator=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            probs = (logits[:, -1, :] / temperature).softmax(-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(idx.device)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
