"""Build the leaderboard table + loss-curve plot from results/*.json and
splice them into README.md between the LEADERBOARD markers.

    python -m bbp.leaderboard
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
BADGE = {"none": "🚫 none", "local": "🧱 local", "modified": "🪞 modified", "full": "✅ full"}


def load():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    return sorted(rows, key=lambda r: r["val_loss"] if math.isfinite(r["val_loss"]) else 1e9)


def table(rows) -> str:
    bp = next((r["val_loss"] for r in rows if r["rule"] == "backprop" and not r["tag"]), None)
    res = next((r["val_loss"] for r in rows if r["rule"] == "reservoir" and not r["tag"]), None)
    out = [
        "| # | rule | autograd | val loss ↓ | vs backprop | beats random reservoir? | steps | fwd passes | tuned |",
        "|---|------|----------|-----------:|------------:|:--:|------:|-----------:|-------|",
    ]
    for i, r in enumerate(rows, 1):
        name = r["rule"] + (f" ({r['tag']})" if r["tag"] else "")
        v = r["val_loss"]
        d = f"{v - bp:+.3f}" if bp is not None and math.isfinite(v) else "—"
        beat = "—" if r["rule"] in ("reservoir",) or res is None else ("yes" if v < res else "no")
        sw = r.get("sweep")
        tuned = f"{sw['key']}={r['hp'][sw['key']]:.2g}" if sw else ""
        out.append(
            f"| {i} | `{name}` | {BADGE[r['autograd']]} | {v:.3f} | {d} | {beat} | {r['steps']:,} | {r['forwards']:,} | {tuned} |"
        )
    if rows:
        r0 = rows[0]
        out.append(
            f"| | *unigram (letter frequencies only)* | | {r0['unigram_loss']:.3f} | | | | | |"
        )
        out.append(f"| | *untrained / uniform guess* | | {r0['init_val']:.3f} | | | | | |")
        out.append("")
        out.append(
            f"<sub>{r0['params']/1e6:.2f}M-param char GPT ({r0['model']['n_layer']} layers, d={r0['model']['n_embd']}, "
            f"ctx {r0['model']['block_size']}), batch {r0['batch_size']}, TinyShakespeare, {r0['budget_minutes']:g} min of "
            f"training time per rule on `{r0['device']}`, val loss in nats/char. Each rule's `tuned` knob got a 3-point sweep first.</sub>"
        )
    return "\n".join(out)


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    for r in rows:
        c = r["curve"]
        ls = {"none": ":", "local": "--", "modified": "-.", "full": "-"}[r["autograd"]]
        ax.plot([p["t"] / 60 for p in c], [p["val"] for p in c], ls, lw=2, label=r["rule"])
    ax.axhline(rows[0]["unigram_loss"], color="grey", lw=1)
    ax.text(0.02, rows[0]["unigram_loss"] + 0.03, "unigram floor", color="grey", fontsize=8)
    ax.set_xlabel("training time (min)")
    ax.set_ylabel("val loss (nats/char)")
    ax.set_title("beyond-backprop: who can teach a tiny GPT Shakespeare?")
    ax.set_ylim(1.3, 4.3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, title="line = autograd use:  — full  -. modified  -- local  ·· none", title_fontsize=7)
    fig.tight_layout()
    out = RESULTS / "curves.png"
    fig.savefig(out)
    return out


def samples(rows) -> str:
    parts = ["<details><summary>What each rule's model writes after its run (prompt <code>ROMEO:</code>)</summary>\n"]
    for r in rows:
        parts.append(f"**{r['rule']}** (val {r['val_loss']:.3f})\n\n```\n{r['sample'][:300]}\n```\n")
    parts.append("</details>")
    return "\n".join(parts)


def main(argv=None):
    rows = load()
    if not rows:
        print("no results yet - run python -m bbp.tournament")
        return
    md = table(rows)
    png = plot(rows)
    block = md + ("\n\n![loss curves](results/curves.png)\n" if png else "") + "\n" + samples(rows)
    print(md)
    if README.exists():
        text = README.read_text()
        if START in text:
            text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
            README.write_text(text)
            print(f"\nupdated {README}")


if __name__ == "__main__":
    main(sys.argv[1:])
