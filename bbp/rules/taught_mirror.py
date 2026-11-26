"""TAUGHT MIRROR: a lookup table teaches a network that never uses backprop.

Two pieces, neither of them backprop:
  1. A counting teacher (no gradients at all). For the first 1,500 steps the rule just
     counts which character follows every recent string in the batches it sees, then
     builds a smoothed 9-gram table (Witten-Bell backoff), leave-one-out so it predicts
     each training character as if it had never seen it.
  2. A mirror_neurons student: backward weights that start random and get the same
     update as their forward twins (Kolen & Pollack), so no weight is ever copied.
The student's loss is 70% the real next character and 30% the table's whole distribution.

The table alone does not fit in the model (~1.4M contexts x 65 next characters), but what
the student learns from it does. Still an end-to-end backward pass with learned stand-in
weights, so 'modified'. Control to read it against: backprop with the same teacher.
"""
from . import register
from .mirror_neurons import MirrorNeurons


@register
class TaughtMirror(MirrorNeurons):
    name = "taught_mirror"
    blurb = "mirror_neurons learning 30% from a smoothed 9-gram table it builds by counting the batches it sees (leave-one-out); no weight copied, no gradient in the teacher."
    paper = "Kolen & Pollack 1994 (mirror); Witten & Bell 1991 (smoothing); Hinton et al. 2015 (soft targets)"
    defaults = dict(MirrorNeurons.defaults, teach=0.3, beta=16.0, order=9, count_steps=1500)
