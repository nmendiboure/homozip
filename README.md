# homozip

A minimal, exactly solvable model of how a broken chromosome finds its
matching copy.

After a double-strand break, a Rad51 filament has to locate the one place
in the genome that matches the broken end. `homozip` keeps a single step of
that search, the sequence test itself, and leaves out everything else: no
nucleus, no contact map, no D-loop. What is left is one small Markov chain
per filament site, and it can be solved with a pen. The point is to say
which properties of the search are decided by counting nucleotides alone,
before any rate or any geometry enters.


## The chain

A filament carries `N` nucleation sites. Each site can catch a piece of
genomic DNA sharing `k = 8` identical nucleotides with it: the homologous
donor with probability `f`, an unrelated locus otherwise. The background
is everywhere, the donor is one locus in a nucleus, within reach of the
filament a fraction `c` of the time. That one number is the only place
where the nucleus enters. The match then
extends one nucleotide at a time. At each step the next nucleotide pairs
with probability `p`, which is `1 - delta` on a donor of divergence
`delta` and `p_het` in the background. A mismatch stops the zipper once
`m` of them have been stepped over; reaching `L_c` nucleotides is the
point of no return.

| what happens | transition | rate |
| --- | --- | --- |
| a site catches a seed | `S -> T[k]` | `kon f c` on the donor, `kon (1-f)` in the background |
| the next nucleotide pairs | `T[L] -> T[L+1]` | `kext p` |
| a mismatch blocks the joint | `T[L] -> Tb[L]` | `kext (1-p)` |
| the joint falls off | `T[L], Tb[L] -> S` | `koff` |

| parameter | value | where it comes from |
| --- | --- | --- |
| `N`, `k_seed`, `L_commit`, `max_mismatches` | 200, 8 nt, 30 nt, 0 | structural |
| `f` | 1.57e-3 | **measured**: share of the LY filament's 8-mer matches that fall on *LYS2* |
| `c` | 0.025 | share of the time the donor is within reach; the one fitted number, set so that half of the cells commit by about 4 h |
| `p_het` | 0.271 | **measured** on S288c, flat over `L = 8..14` |
| `delta` | 0 | the variable of the experiment |
| `kon`, `koff0`, `kext` | 1, 0.7 min⁻¹, 300 nt min⁻¹ | `kext` sets the unit; `kon` puts 60 % of the sites to work at once |

`params.yaml` carries the same list with comments.


## What it says

The three things a joint can do have rates adding up to `kext + koff`,
which does not contain `p`: the sequence decides which branch is taken,
never how long the joint waits. Everything follows from that.

- **Specificity is counted, not timed.**
  `P_commit = Q * Prob[Binomial(n, 1-p) <= m]` with `Q = (kext/(kext+koff))^n`
  and `n = L_c - k`. `Q` is the same on both tracks, so every ratio of
  commitment probabilities is free of rates: a donor is accepted
  `(1/p_het)^22 = 3e12` times more readily than a random locus, whatever
  the filament stability. (`divergence.pdf`)
- **The commitment length is derived.** Keeping false commitments below a
  fraction `eps` of the true ones needs
  `n >= ln((1-f)/(f c eps)) / ln(p_hom/p_het)`. With the donor always
  within reach (`c = 1`) that is 17, 19 and 20 nt for one false
  commitment per hundred, per thousand and per ten thousand, a statement
  about the sequence alone; a donor within reach 2.5 % of the time costs
  three more nucleotides, 20, 22 and 23. Length costs time only on a
  divergent donor. (`fidelity.pdf`, `commitment_length.pdf`)
- **Rates set the load.** `commit_rate = kon f c S_free P_commit` with
  `S_free = N / (1 + kon tau)`. A background seed blocks after about
  1.4 nt and holds its site for `1/koff`, so the rate saturates with
  `kon`, the filament fills, and filament stability has an optimum that is
  broad on an empty filament and sharp on a crowded one. At `kon = 1` per
  site 60 % of the sites hold a joint at any time: the filament tests many
  segments at once, the intersegmental search of Forget and
  Kowalczykowski (2012), and more time is lost on background joints than
  waiting for the donor. (`association.pdf`, `occupancy.pdf`,
  `stability.pdf`)
- **The sequence test is not the bottleneck.** With the measured `f`, a
  busy filament and the donor always within reach, the first donor
  commitment would come after 8 min. The hours measured in cells are the
  factor `c`: half of the cells commit at 3.7 h with `c = 0.025`. What
  `c` stands for, filament mobility and chromosome contacts, is the whole
  business of a spatial model; here it is one number.
- **The first commitment is memoryless.** A site forgets its state
  within one transit of a few seconds and one blocked joint of a minute,
  so the time to the first donor commitment is exponential from `t = 0`.
  Measured search times are peaked and the DLE curve is sigmoidal: that
  shape has to come from what this model holds constant, a `c` that rises
  as resection frees the filament, or a delay before the search starts.
  (`first_passage.pdf`, `hazard.pdf`)
- **`p_het` and `f` survive a real genome.** Every 8-mer of the LY
  filament, matched against S288c and extended to its first mismatch,
  gives a flat `p(L) = 0.27` up to 14 nt and a donor plateau at `f`.
  Repeats raise the background tail above 15 nt by a factor thirty at
  30 nt, against a discrimination budget of 1e12.
  (`spectrum_survival.pdf`, `spectrum_hazard.pdf`)

The derivations are in the docstrings of `model.py`, in the same order.


## Use

```bash
conda env create -f environment.yml && conda activate homozip   # or pip install -e .

homozip theory   params.yaml                  # closed forms and the fidelity bound
homozip figures  params.yaml -o output/fig    # one PDF per panel
homozip spectrum --genome S288c.fa --filament LY.yaml \
                 [--mask chr2:471858-473926] --output output/spectrum.json
```

`resources/spectrum/` holds the two measurements behind `f` and `p_het`,
one on the whole genome and one with the donor masked.

```
src/homozip/model.py      parameters, the one-site chain, and its solution
src/homozip/genome.py     match-length spectrum of a filament against a genome
src/homozip/figures.py    one PDF per panel
src/homozip/main.py       command line
tests/test_homozip.py     the closed forms against the chain
```


## References

Bitran et al., *PLoS Comput Biol* 13:e1005421 (2017) ·
Forget & Kowalczykowski, *Nature* 482:423 (2012) ·
Fujitani, Yamamoto & Kobayashi, *Genetics* 140:797 (1995) ·
Savir & Tlusty, *Mol Cell* 40:388 (2010) ·
Wiktor et al., *Nature* 597:426 (2021).
