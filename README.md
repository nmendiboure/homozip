# homozip

A minimal model of how a broken chromosome finds its matching copy.

After a double-strand break, a RAD51 filament has to locate the one place
in the genome that matches the broken end. `homozip` keeps a single step of
that search, the sequence test itself, and leaves out everything else. The
model is small enough to solve on paper, and the stochastic simulation is
there to check the algebra rather than to stand in for it.

Two of its numbers are fitted, two are measured on a real genome, the rest
are structural.


## How it works

A filament carries `N` nucleation sites. Each one can catch a piece of
genomic DNA that shares `k_seed = 8` identical nucleotides with it. That
piece is the homologous donor with probability `f`, and an unrelated locus
the rest of the time.

The match then extends one nucleotide at a time, like a zipper. At each
step the next nucleotide pairs with probability `p`:

| track | `p` |
| --- | --- |
| the homologous donor | `1 - delta`, with `delta` the sequence divergence |
| the genome background | `0.271`, measured on the yeast genome |

A mismatch stops the zipper. Reaching `L_commit = 30` nucleotides is the
point of no return.

| what happens | reaction | rate |
| --- | --- | --- |
| a site catches a seed | `S -> T[k, 0]` | `kon f` on the donor, `kon (1-f)` in the background |
| the next nucleotide pairs | `T[L, j] -> T[L+1, j]` | `kext p` |
| a mismatch is stepped over | `T[L, j] -> T[L+1, j+1]`, while `j < m` | `kext (1-p)` |
| a mismatch blocks the joint | `T[L, m] -> Tb[L]` | `kext (1-p)` |
| the joint falls off | `T[L, j] -> S`, `Tb[L] -> S` | `koff(L)` |

A blocked joint cannot grow any more, but it is not stuck: it still falls
off at `koff`. With the default settings this is 91 species and 178
reactions, one state per nucleotide, with no length binning anywhere.

Every reaction is first order and the sites share nothing, so the model is
`N` independent copies of one small Markov chain. That is what makes the
search time exactly solvable.


## Parameters

`params.yaml` holds the same list with the same comments.

### Structure of the search

| parameter | what it does | value | where it comes from |
| --- | --- | --- | --- |
| `N` | nucleation sites on the filament | 200 | structural |
| `k_seed` | length of the initial exact match | 8 nt | structural, the shortest microhomology a nucleation can hold |
| `L_commit` | length at which the joint is committed | 30 nt | structural, one threshold in place of the whole post-synaptic cascade |
| `max_mismatches` | mismatches the zipper steps over before blocking | 0 | structural, `0` means every mismatch blocks |
| `f` | share of seeds landing on the donor | 1.57e-3 | **measured**, 2040 donor pairs out of 1 295 385 |

### Rates, in events per minute

| parameter | what it does | value | where it comes from |
| --- | --- | --- | --- |
| `kon` | a free site catches a seed | 0.006 per site | **fitted**, so `kon N = 1.2` per minute for the whole filament |
| `koff0` | a joint falls off the DNA | 0.7 | **fitted** |
| `kext` | the zipper advances by one nucleotide | 300 nt/min | fixed. It sets the time unit, and zipping is already fast next to `koff0` |
| `lam` | decay length of `koff(L) = koff0 exp(-(L-k)/lam)` | `null`, a flat `koff` | measured to be worth 3.2 %, so left out. Set `lam: 8.41` to switch it on |

### Sequence

| parameter | what it does | value | where it comes from |
| --- | --- | --- | --- |
| `delta` | divergence of the donor, so `p_hom = 1 - delta` | 0 | the variable of the experiment. 0 is isogenic, 0.01 to 0.10 is the homeologous series |
| `p_het` | chance that the next nucleotide pairs in the background | 0.271 | **measured** on S288c, flat over `L = 8..14` with the donor masked |

### Simulation

| parameter | what it does | value |
| --- | --- | --- |
| `t_end` | length of a run, minutes | 480 |
| `n_points` | output grid points | 481 |
| `n_cells` | independent runs | 2000 |
| `seed` | first random seed, cell *i* uses `seed + i` | 1999 |


## What comes out of it

**A donor is accepted 3e12 times more readily than a random locus, and no
rate is involved.** At any length, the three things that can happen have
rates adding up to `kext + koff`, which does not contain `p`. The sequence
changes which branch is taken, never how long it takes, so every ratio of
commit probabilities is free of kinetics. Making the filament more or less
stable cannot buy specificity.

**The decline of recombination with divergence comes for free, and its
slope is a measurement.** With `p = 1 - delta` the yield goes as
`exp(-n delta)`, where `n = L_commit - k_seed`. A test 18 nt long and one
80 nt long predict slopes of 10 and 72, which a divergence series tells
apart at a glance.

**The commitment length can be derived rather than fitted.** Background
seeds outnumber donor seeds by `(1 - f)/f` to one, so keeping false
commitments rare sets a lower bound on the length of the test. With the
measured `f` and `p_het`, one false commitment per thousand needs 19 nt.
Thresholds in that range are usually fitted to recombination data.

**The search time is exactly solvable, and that is also the model's most
useful failure.** The sites are independent, so the time to the first
commitment is the minimum of `N` identical variables and costs one matrix
exponential. It comes out exponential, while Wiktor et al. measure a
peaked search time in *E. coli*. No memoryless mechanism gives a peak, so
it has to come from something this model throws away: filament mobility,
chromosome proximity, or a delay before the search starts.

The reasoning behind each of these, with the derivations, is in
[`docs/homology-zipper.md`](docs/homology-zipper.md).


## What it leaves out

No Hi-C weighting, no filament mobility, no D-loop geometry, no second
proofreading stage, no resection delay, no genome index. Each one is
either downstream of commitment, or acts the same way on both tracks, or
enters only through `f`.


## Install

```bash
conda env create -f environment.yml && conda activate homozip
```

or `pip install -e .` in any Python 3.12 environment.


## Use

```bash
homozip theory  params.yaml                 # closed forms and the fidelity bound
homozip build   params.yaml -o output       # Antimony source and SBML
homozip run     params.yaml -o output       # simulation against the exact solution
homozip figure  params.yaml -o output       # the figures; --quick for a smoke run
homozip spectrum --genome S288c.fa --filament LY.yaml \
                 [--mask chr2:471858-473926] --output output/spectrum.json
```

`resources/spectrum/` holds the two measurements behind `f` and `p_het`,
one on the whole genome and one with the donor masked, so the figures can
be redrawn without the genome at hand.


## Layout

```
src/homozip/model.py      parameters, reaction network, Antimony and SBML export
src/homozip/theory.py     closed forms and the exact first-passage solution
src/homozip/simulate.py   Gillespie runs with tellurium
src/homozip/genome.py     match-length spectrum of a filament against a genome
src/homozip/figures.py    the figures
src/homozip/main.py       command line
params.yaml               the parameters, with their provenance
docs/homology-zipper.md   the reasoning
tests/                    the closed forms, and the simulation against the exact solution
```


## References

Bitran et al., *PLoS Comput Biol* 13:e1005421 (2017) ·
Fujitani, Yamamoto & Kobayashi, *Genetics* 140:797 (1995) ·
Kochugaeva, Shvets & Kolomeisky, *Biophys J* 112:859 (2017) ·
Savir & Tlusty, *Mol Cell* 40:388 (2010) ·
Wiktor et al., *Nature* 597:426 (2021).
