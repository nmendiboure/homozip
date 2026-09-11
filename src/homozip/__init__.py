"""
homozip: the mismatch-limited zipper.

A minimal stochastic model of homology search after a double-strand break.
A microhomology seed extends one nucleotide at a time until it reaches a
commitment length or falls off; the sequence enters through a single
number, the probability that the next nucleotide pairs.

    model     parameters, reaction network, Antimony / SBML export
    theory    closed forms and the exact first-passage solution
    simulate  Gillespie runs (tellurium)
    genome    measures p_het and f on a real genome
    figures   the figures
    cli       homozip build | theory | run | spectrum | figure
"""

__version__ = "1.0.0.dev0"
