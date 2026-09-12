"""
homozip: the mismatch-limited zipper.

A minimal, exactly solvable model of homology search after a double-strand
break. A microhomology seed extends one nucleotide at a time until it
reaches a commitment length or falls off; the sequence enters through a
single number, the probability that the next nucleotide pairs.

    model     parameters, the one-site chain, and its solution
    genome    measures p_het and f on a real genome
    figures   one PDF per panel
    main      homozip theory | figures | spectrum
"""

__version__ = "1.0.0.dev0"
