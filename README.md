# homozip

The mismatch-limited zipper: a minimal stochastic model of how a RAD51
filament finds its homologous donor after a double-strand break.

It deliberately does very little. It keeps one step, the sequence test
itself, and keeps it small enough to solve with a pen. The Gillespie
simulation then checks the algebra rather than being the only source of
truth, and the model exports to SBML so it can be run anywhere.

Two of its numbers are fitted. Two are measured on a real genome. The rest
are structural or swept.


## The model

A filament carries `N` nucleation sites. Each one can catch a piece of
genomic DNA with which it shares `k_seed = 8` identical nucleotides. That
piece is the homologous donor with probability `f`, and some unrelated
locus the rest of the time.

The match then extends one nucleotide at a time, like a zipper. At each
step the next nucleotide of the donor pairs with probability `p`. On the
homologous donor `p = 1 - delta`, where `delta` is the sequence
divergence. In the genome background `p = 0.271`, measured on S288c. This
single number per track is the entire sequence layer of the model.

| what happens | reaction | rate |
| --- | --- | --- |
| a site catches a seed | `S -> T[k, 0]` | `kon f` on the donor, `kon (1-f)` in the background |
| the next nucleotide pairs | `T[L, j] -> T[L+1, j]` | `kext p` |
| a mismatch is stepped over | `T[L, j] -> T[L+1, j+1]`, while `j < m` | `kext (1-p)` |
| a mismatch blocks the joint | `T[L, m] -> Tb[L]` | `kext (1-p)` |
| the joint falls off | `T[L, j] -> S`, `Tb[L] -> S` | `koff(L)` |

Reaching `L_commit` nucleotides is the point of no return. A blocked joint
cannot grow any more, but it is not stuck forever: it still falls off at
`koff`. With the default settings this is 91 species and 178 reactions,
one species per nucleotide, with no length binning anywhere.

Every reaction is first order and the sites share nothing, so the model is
`N` independent copies of one small Markov chain. That is what makes the
search time exactly solvable.


## Parameters

Everything the model contains, and where each number comes from.
`params.yaml` holds the same list with the same comments.

### Structure of the search

| parameter | what it does | value | where it comes from |
| --- | --- | --- | --- |
| `N` | nucleation sites carried by the filament | 200 | structural |
| `k_seed` | length of the initial exact match | 8 nt | structural, the shortest microhomology a nucleation can hold |
| `L_commit` | length at which the joint is committed | 30 nt | structural. One threshold in place of the whole post-synaptic cascade. Only `L_commit - k_seed` can be identified, and a divergence series measures it |
| `max_mismatches` | mismatches the zipper steps over before blocking | 0 | structural. `0` means every mismatch blocks |
| `f` | share of seeds landing on the homologous donor | 1.57e-3 | **measured**: the plateau of the match-length survival curve of the LY filament against S288c, 2040 donor pairs out of 1 295 385 |

### Rates, in events per minute

| parameter | what it does | value | where it comes from |
| --- | --- | --- | --- |
| `kon` | a free site catches a seed | 0.006 per site | **fitted**. `kon N = 1.2` per minute is the nucleation rate of the whole filament |
| `koff0` | a joint falls off the DNA, at the seed length | 0.7 | **fitted** |
| `kext` | the zipper advances by one nucleotide | 300 nt/min | fixed, not fitted. Doubling it moves the commit rate by 2.6 % and making it infinite by 5.3 %, because zipping is already effectively instantaneous next to `koff0`. It sets the time unit: only `koff0/kext` and `kon/kext` carry content |
| `lam` | decay length of `koff(L) = koff0 exp(-(L-k)/lam)` | `null`, meaning a flat `koff` | a modelling choice, and a conclusion rather than an assumption. The ladder was implemented and measured: it changes the commit rate by 3.2 %. Set `lam: 8.41` for a joint stabilised by a factor 0.7 per helical triplet |

### Sequence

| parameter | what it does | value | where it comes from |
| --- | --- | --- | --- |
| `delta` | divergence of the homologous donor, so `p_hom = 1 - delta` | 0 by default | this is the variable of the experiment. 0 is isogenic; 0.01 to 0.10 covers the classical homeologous recombination series |
| `p_het` | probability that the next nucleotide pairs in the genome background | 0.271 | **measured**, flat over `L = 8..14` with the donor masked. It is not 1/4: the yeast genome is AT-rich and the sum of squared base frequencies at 38 % GC is 0.264. A measured profile `p(8), p(9), ...` can be given instead of a scalar |

### Simulation

| parameter | what it does | value |
| --- | --- | --- |
| `t_end` | length of a run, minutes | 480, i.e. 8 hours |
| `n_points` | output grid points | 481 |
| `n_cells` | independent runs | 2000 |
| `seed` | first random seed; cell *i* uses `seed + i` | 1999 |


## What comes out of it

**Specificity is combinatorial, and no rate takes part in it.** At each
length the three things that can happen have rates summing to
`kext + koff`, which does not contain `p`. The time spent at a given
length is therefore the same whatever the sequence, and only the branch
taken differs. The chance that one seed commits is

```
P_commit = Q * Prob[ Binomial(n, 1 - p) <= m ]      with Q = (kext / (kext + koff0))^n
         = p^n Q                                     when m = 0,  n = L_commit - k_seed
```

`Q` is the only place the rates enter, and `Q` is the same on both tracks.
Any ratio of commit probabilities is therefore free of kinetics. With the
default settings, `n = 22`, `Q = 0.95`, and a donor is `3e12` times more
likely to be accepted than a random locus. A length-dependent
stabilisation of the filament cannot contribute to that, whatever else it
does.

**The exponential decline of homeologous recombination comes for free, and
its slope is a measurement.** With `p = 1 - delta`, the yield goes as
`exp(-n delta)`, so the slope of a divergence series measures
`L_commit - k_seed` directly. A threshold of 18 nt and one of 80 nt
predict slopes of 10 and 72, which a divergence series tells apart at a
glance. If mismatches are tolerated, the curve is flat at low divergence and
bends into the exponential later; the width of that shoulder measures `m`.

**The commitment length can be derived instead of fitted.** Background
seeds outnumber donor seeds `(1 - f) / f` to one, so keeping false
commitments below a fraction `eps` of the true ones requires

```
n >= ln( (1 - f) / (f eps) ) / ln( p_hom / p_het )
```

which contains no rate and nothing fitted. With the measured `f` and
`p_het` this gives `L_commit >= 17` at one false commitment per hundred,
19 per thousand, 20 per ten thousand. Commitment thresholds in that range
are commonly fitted to recombination data; here the range is derived.

**The search time is exactly solvable.** Because the sites are
independent, the time to the first donor commitment is the minimum of `N`
identical phase-type variables, computed with one matrix exponential. The
single-site hazard rises over one transit time, about four seconds, and is
flat after that: the search has no memory, and the first-commitment time
is exponential. On 10 000 Gillespie cells the empirical and exact
distribution functions differ by 0.006, against a 5 % critical value of
0.014.

That last result is also the model's most useful failure. Wiktor et al.
measure a peaked, gamma-shaped search time in *E. coli*. No memoryless
mechanism produces that shape, so a peak has to come either from steps
before the test, such as resection, which is a pure time offset, or from a
search that accumulates progress. Filament mobility and chromosome
proximity are exactly what this model throws away.


## What it leaves out

No Hi-C weighting, no filament mobility, no D-loop geometry, no second
proofreading stage, no resection delay, no genome index. Each of those is
either downstream of commitment, or acts identically on both tracks, or
enters only through `f`. The reasoning for each is in
[`docs/homology-zipper.md`](docs/homology-zipper.md), along with the full
derivations.


## Install

```bash
conda env create -f environment.yml && conda activate homozip
```

or `pip install -e .` in any Python 3.12 environment.

## Use

```bash
homozip theory  params.yaml                 # closed forms, fidelity bound, ladder gain
homozip build   params.yaml -o output       # Antimony source and SBML
homozip run     params.yaml -o output       # Gillespie ensemble against the exact solution
homozip figure  params.yaml -o output       # the figures; --quick for a smoke run
homozip spectrum --genome S288c.fa --filament Lys2-LY.yaml \
                 [--mask chr2:471858-473926] --output output/spectrum.json
```

`resources/spectrum/` holds the two measurements behind `f` and `p_het`,
one on the whole genome and one with the donor masked, so the figures can
be redrawn without the genome at hand.

## Layout

```
src/homozip/model.py      parameters, reaction network, Antimony and SBML export
src/homozip/theory.py     closed forms, and the exact first-passage solution
src/homozip/simulate.py   Gillespie runs with tellurium
src/homozip/genome.py     match-length spectrum of a filament against a genome
src/homozip/figures.py    the figures
src/homozip/cli.py        command line
params.yaml               the parameters, with their provenance
docs/homology-zipper.md   the reasoning
tests/                    identities of the closed forms, and the SSA against the exact solution
```

## References

Bitran et al., *PLoS Comput Biol* 13:e1005421 (2017) ·
Fujitani, Yamamoto & Kobayashi, *Genetics* 140:797 (1995) ·
Kochugaeva, Shvets & Kolomeisky, *Biophys J* 112:859 (2017) ·
Savir & Tlusty, *Mol Cell* 40:388 (2010) ·
Wiktor et al., *Nature* 597:426 (2021).
