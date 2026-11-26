"""BACKDROP: plain backprop, except the error is dropped for a random fraction of token
positions every time it crosses a block boundary on the way down. The forward pass and
the model are untouched (unlike dropout), only the learning signal is thinned out.
Not the true gradient, so 'modified'."""
from . import register
from ._backdrop import install_backdrop
from .backprop import Backprop


@register
class Backdrop(Backprop):
    name = "backdrop"
    autograd = "modified"
    blurb = "Backprop whose error signal is randomly dropped per token at every block boundary on the way down; the forward pass is untouched."
    paper = "Golkar & Cranmer 2018, 'Backdrop: stochastic backpropagation'"
    defaults = dict(Backprop.defaults, p=0.25)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        install_backdrop(model, self.hp["p"])
