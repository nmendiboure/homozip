# The mismatch-limited zipper

Why the model is built this way, what can be derived from it, and what it
cannot say. The README describes the model itself and lists the
parameters.

Code: [`model.py`](../src/homozip/model.py),
[`theory.py`](../src/homozip/theory.py),
[`simulate.py`](../src/homozip/simulate.py),
[`genome.py`](../src/homozip/genome.py),
[`figures.py`](../src/homozip/figures.py).


## 1. The two choices that carry everything

**A block is permanent.** A mismatch belongs to the donor sequence, not to
the attempt. The base at `L+1` either pairs or it does not, and trying
again will not change it. That is what makes the sequence a fixed property
of the locus instead of a coin flipped at every step. A blocked joint is
not stuck forever, though: it falls off at `koff` like any other.

**`koff` does not depend on length.** The ladder
`koff0 exp(-(L - k)/lam)` is implemented and can be switched on, but it
moves the commit rate by 3.2 %, so it is left out. That is a result rather
than an assumption, and section 5 says why the number is so small.


## 2. Specificity comes from counting, not from rates

A joint at length `L` can do three things, and their rates add up to

```
kext p + kext (1 - p) + koff(L) = kext + koff(L)
```

which does not contain `p`. The time spent at a given length is the same
whatever the sequence; only the branch taken differs. Writing
`s = kext / (kext + koff)` for the chance of surviving one step,

```
P_commit = Q * Prob[ Binomial(n, 1 - p) <= m ],   Q = s^n,   n = L_commit - k_seed
```

`Q` is the only place the rates enter, and it is the same on both tracks.
Any ratio `P_commit(p1) / P_commit(p2)` therefore contains no rate at all.

Two consequences.

**No stabilisation law can contribute to specificity**, since it acts
identically on the donor and on the background. Specificity comes from the
number of consecutive checks demanded and from nothing else. At the
default settings `n = 22`, `Q = 0.950`, and the discrimination is
`(1/0.271)^22 = 3.0e12`.

**A divergent donor is not slower to zip, only less likely to finish.**
The transit time conditional on success is `sum_L 1/(kext + koff(L))`,
which is also free of `p`. Divergence should move the yield of
recombination without moving the timing of the events that do succeed,
which is a signature worth looking for.


## 3. Divergence: the slope measures the length of the test

With `p = 1 - delta` and no tolerance,

```
P_commit(delta) = (1 - delta)^n Q  ~  Q exp(-n delta)
```

so the slope of `log(yield)` against divergence is exactly
`n = L_commit - k_seed`. The exponential decline of homeologous
recombination is the model's first-order behaviour, not evidence for a
mismatch-repair activity of a given strength. A threshold of 18 nt and one
of 80 nt predict slopes of 10 and 72; fitting the isogenic curve cannot
tell them apart, a divergence series can.

With `m > 0` the law acquires a shoulder. `Prob[Binomial(n, delta) <= m]`
is flat while `n delta` is small next to `m`, then recovers
`exp(-n delta)`. The width of the shoulder measures `m`, the slope past it
still measures `n` (figure, panel A).

Fujitani et al. read the minimum efficient processing segment differently,
as `2/sqrt(h)`, on a fully homologous tract of varying length. That is
post-synaptic branch migration. This is a divergence series at fixed
length, which is the presynaptic test. `L_commit` here is a threshold on
checks rather than on bases, and that is what makes it measurable by
divergence.


## 4. How long the test has to be

Nothing so far says why a commitment threshold has to exist. This does.
Background seeds outnumber donor seeds `(1 - f)/f` to one, and each is
`(p_het/p_hom)^n` less likely to get through, so the odds of a false
commitment are `((1 - f)/f) (p_het/p_hom)^n`. Keeping them below a
tolerance `eps` gives a lower bound with no kinetics and nothing fitted in
it:

```
n >= ln( (1 - f) / (f eps) ) / ln( p_hom / p_het )
```

With the measured `f = 1.57e-3` and `p_het = 0.271`, on an isogenic donor:

| tolerated false rate | `n_min` | `L_commit >= k + n_min` |
| --- | --- | --- |
| 1e-2 | 8.5 | 17 |
| 1e-3 | 10.2 | 19 |
| 1e-4 | 12.0 | 20 |

Commitment thresholds around 18 nt are commonly fitted to recombination
data. Here the range is derived instead. Read the other way, a threshold
of 18 nt admits 1.4e-3 false joints per true one, about one per thousand.

The bound also shows the shape of the problem. `n_min` grows only
logarithmically in the required fidelity, so three more orders of
magnitude of specificity cost about five nucleotides. Recognition is cheap
once there is a per-nucleotide check. Speed is not.

Panel C draws the trade-off. On an isogenic donor the search time barely
moves as `L_commit` grows, while the false-commitment odds fall by
twenty-seven decades. On a donor 5 % divergent the search time doubles
every fourteen nucleotides or so, and the threshold becomes a real design
variable.


## 5. The search time, exactly

The `N` sites are independent, so

```
T_search = min(T_1, ..., T_N),   F_N(t) = 1 - (1 - F_1(t))^N
```

and `F_1`, the absorption of one site into the committed state, is a
phase-type distribution on the 91-state chain built from the same reaction
list that produces the Antimony source. On a uniform grid it costs one
matrix exponential per time point, with no ensemble and no approximation
(`theory.search_time_distribution`).

What it shows, in panel D:

| | |
| --- | --- |
| single-site hazard | rises from zero over one transit time, about four seconds, then flat |
| exact mean search time | 563.55 min, against 563.57 min for the steady-state estimate |
| simulation against exact | Kolmogorov distance 0.006 on 10 000 cells, 5 % critical value 0.014 |
| committed by 480 min | 57.5 ± 0.5 % simulated, 57.3 % exact |

Once the hazard is flat, `N h_1` equals `kon f S_free P_commit` to two
parts in a thousand, so the steady-state rate used in sections 2 to 4 is
the exact asymptotic hazard.

**Where the time goes.** Little's law splits the mean search time into the
number of donor encounters needed times the cost of one round:

```
1 / commit_rate = (1 / P_commit) [ 1/(kon f) + ((1-f)/f) tau_het + tau_hom ] / N
```

| | time |
| --- | --- |
| waiting for a donor to be caught | 1.1e5 min |
| lost on background loci in between | 9.1e2 min |
| zipping up the donor itself | 0.07 min |

The search is limited by nucleation, not by off-targets, by two orders of
magnitude. That is why `koff0` hardly matters here.

**This is also the clearest statement of what the model cannot produce.**
It predicts an exponential first-commitment time after a transient of a
few seconds. Wiktor et al. measure, in *E. coli*, a search time with a
mean of 9 min and a standard deviation of 3 min, which is strongly peaked.
No memoryless mechanism gives that shape. A peak has to come either from
steps before the test, such as resection and filament assembly, which are
a pure time offset and can be convolved in, or from a search that
accumulates progress: filament repositioning, reduced dimensionality,
local enrichment by chromosome conformation.


## 6. Clogging, and the best filament stability

A background seed blocks after about `1/(1 - p_het)` nucleotides and then
holds its site for `1/koff0` rather than `1/kext`. The two occupancy times
are 1.43 min for a background seed and 0.07 min for a donor seed. Little's
law gives `S_free = N / (1 + kon tau_eff)` and
`commit_rate = kon f S_free P_commit`.

Both limits of `koff0` are bad. Too stable and the filament fills with
dead background joints; too labile and donor seeds are released mid-zip.
So there is an interior optimum,

```
koff0* ~ sqrt( (1 - p_het) kon kext / lam_eff )
```

a geometric mean of the nucleation and zipping rates. The fitted
`koff0 = 0.7` sits a factor three above it, at 98 % of the attainable
rate.

That agreement should be reported carefully. At this nucleation load the
optimum is broad, with 1.3 decades of `koff0` staying within 5 % of the
maximum (panel B). It is not evidence that filament stability was tuned
for search speed. It is evidence that in the dilute regime the search is
insensitive to filament stability altogether. Push the load up by a factor
300 and the optimum sharpens, `S_free` falls to 56 out of 200, and the
same value delivers half the attainable rate. Filament stability becomes a
design variable only once the filament is crowded.


## 7. Does a constant `p` survive a real genome?

`p = 1/4` is the random-sequence null, and a real genome is not random.
`homozip spectrum` measures the real thing: for every 8-mer of the LY
filament, every exact match in S288c extended to its first mismatch. That
is 1 295 385 (site, locus) pairs. Run it twice, once with the LYS2 donor
masked, and the two effects separate cleanly. The measurements are kept in
`resources/spectrum/` and drawn in `spectrum.png`.

| | measured |
| --- | --- |
| background `p(L)`, L = 8..14 | flat at 0.27 (0.269 to 0.284, mean 0.274) |
| background `p(L)`, L = 16..21 | rises to about 0.35, real repeats |
| background pairs reaching L = 24 | 0 out of 1 293 066 |
| donor plateau, which is `f` | 1.57e-3 |
| whole-genome `p(L)` | 0.27 rising to 1.0, crossing at L ≈ 13 |

Three readings, only one of which is a correction to the model.

**`p_het` is 0.271, and it is a measurement.** S288c is about 38 % GC, so
the chance that two random positions carry the same base is
`2(0.31^2) + 2(0.19^2) = 0.264`. The measured value is that plus a little
short-range compositional correlation. The real result is the flatness:
over seven consecutive nucleotides the background has no memory.

**The rise of the whole-genome `p(L)` to 1.0 is not a missing
ingredient.** Loci that have already matched 13 nt are overwhelmingly the
donor, because the background has been thinned by `0.27^5`, so the
surviving population stops being random. `S(L) = (1 - f) p_het^(L-k) + f`
reproduces the measured curve with nothing added, and the crossover at
L ≈ 13 sits where the bound of section 4 put the threshold. This is also
the answer to the objection that `p` ought to grow as the match gets
longer: it does in the aggregate, and the two-track model already predicts
it.

**The genuine correction is the repeat tail.** Above L ≈ 15 the
donor-masked hazard rises to about 0.35 because of Ty elements,
paralogues and subtelomeric repeats. That makes the background about
thirty times more permissive at L = 30 than the flat law, against a
discrimination budget of 1e12, so the conclusions above do not move. The
generator takes the measured curve directly:

```python
from homozip import genome
from homozip.model import Params
bg = genome.load("resources/spectrum/background.json")
prm = Params(p_het=genome.p_profile(bg, 8, 30))
```

One caveat: above L ≈ 16 the masked spectrum rests on a few tens of pairs
from a single 2 kb filament, so the tail is not well determined. Measuring
it properly needs many filaments, or the genome against itself.


## 8. What is left out, and what it costs

| left out | consequence | why it is acceptable here |
| --- | --- | --- |
| Hi-C weighting | `f` is a scalar, no 4C profile | the rate is linear in `f`, so a structured `f` reweights donors without touching the mechanism |
| filament mobility | `f` is constant in time | the search is memoryless, so under fast switching only the time-averaged `f` enters |
| Rad54, roadblocks, D-loop geometry | no D-loop size histogram | downstream of commitment, and identical on both tracks |
| genome and k-mer index | no per-locus prediction | replaced by `p` and `f`, the only two things the mechanism reads |
| Gamma resection delay | the search starts at t = 0 | a pure time offset; convolve the exact density with the Gamma to compare |
| second proofreading stage | one gate instead of two | two gates multiply, so a divergence slope alone cannot separate them |
| deterministic zipping | Poisson zipping | the mean is matched, the variance is not, which is harmless while `kext` is far above `koff0` |


## References

- Bitran A, Chiang W-Y, Levine E, Prentiss M (2017). Mechanisms of fast
  and stringent search in homologous pairing of double-stranded DNA.
  *PLoS Comput Biol* 13:e1005421.
- Fujitani Y, Yamamoto K, Kobayashi I (1995). Dependence of frequency of
  homologous recombination on the homology length. *Genetics* 140:797.
- Kochugaeva MP, Shvets AA, Kolomeisky AB (2017). On the mechanism of
  homology search by RecA protein filaments. *Biophys J* 112:859.
- Savir Y, Tlusty T (2010). RecA-mediated homology search as a nearly
  optimal signal detection system. *Mol Cell* 40:388.
- Wiktor J, Gynnå AH, Leroy P, Larsson J, Coceano G, Testa I, Elf J
  (2021). RecA finds homologous DNA by reduced dimensionality search.
  *Nature* 597:426.
