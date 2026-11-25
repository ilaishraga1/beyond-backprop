"""Optional extras for gradient-based rules, OFF by default: learning-rate halving every
`half` steps, and replacing a fraction `noise` of the input characters with random ones
(targets stay real). Used to compare rules fairly when overfitting is the wall."""
import torch


def regularise(rule, opts, x):
    hp = rule.hp
    if hp.get("half"):
        k = rule.k = getattr(rule, "k", -1) + 1
        for o in opts:
            for g in o.param_groups:
                g["lr"] = hp["lr"] * 0.5 ** (k / hp["half"])
    if hp.get("noise"):
        x = torch.where(torch.rand(x.shape, device=x.device) < hp["noise"], torch.randint_like(x, rule.model.cfg.vocab_size), x)
    return x
