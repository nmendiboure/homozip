# The mismatch-limited zipper

The reasoning behind `homozip`. The README describes the model and lists
the parameters; this document says why it is built that way, what can be
derived from it, and what it cannot say.

Code: [`model.py`](../src/homozip/model.py),
[`theory.py`](../src/homozip/theory.py),
[`simulate.py`](../src/homozip/simulate.py),
[`genome.py`](../src/homozip/genome.py),
[`figures.py`](../src/homozip/figures.py).


## 1. Why a second, smaller model

An earlier version of this repository transcribed SHERPA reaction by
reaction: `kon`, `koff1(L)`, `kext`, `kdloop(L)`, `koff2(L)`, `kre`, on a
hand-picked grid of twelve lengths. That is fourteen kinetic parameters
and twelve bucket boundaries, and it copied SHERPA's topology without its
data. SHERPA earns its parameters from a real genome, a real Hi-C map and
a real sequence. The transcription had none of those, so the same numbers
ended up constrained only by aggregated D-loop curves. A model in that
position can reproduce a mechanism. It cannot confirm one.

This model was built the other way round. It keeps the microhomology test
and nothing else, and it is small enough to solve on paper, so the
simulation checks the algebra instead of standing in for it.

Its analytical stance is borrowed. Fujitani, Yamamoto and Kobayashi (1995)
formulate a homology test as a random walk with an exact solution.
Kochugaeva, Shvets and Kolomeisky (2017) treat the RecA search as a
discrete-state first-passage problem. Bitran, Chiang, Levine and Prentiss
(2017) frame the speed, stability and stringency trade-off and decompose
the search time. Savir and Tlusty (2010) read the sequence test as a
detection problem with an optimal operating point. Wiktor et al. (2021)
supply the one in-vivo measurement of a search time to hold it against.

| | SHERPA transcription | zipper |
| --- | --- | --- |
| kinetic parameters | 14 | 3 |
| of which fitted | 14 | 2 |
| length discretisation | 12 buckets | none, one state per nucleotide |
| sequence layer | absent | one number per track, measured |
| mismatch handling | an `eps_mm` gate | explicit, block or step over |
| exact solution | none | complete, including the first-passage law |
| species / reactions | 50 / 121 | 91 / 178 |

More states, fewer parameters. The states are free, since they are a
resolution choice with no knob attached. The parameters are not.


## 2. The two decisions that carry everything

**A block is permanent.** A mismatch is a property of the donor sequence,
not of the attempt. The base at `L+1` either pairs or it does not, and
trying again will not change it. The blocked state is what makes the
sequence quenched rather than resampled at every step. A blocked joint is
not stuck forever, though: it falls off at `koff` like any other.

**`koff` is flat.** The length-dependent ladder
`koff0 exp(-(L - k)/lam)` is implemented and can be switched on, but it
moves the commit rate by 3.2 % at this calibration, so it is not part of
the model. That is a result, not an assumption made up front, and section
6 explains why the number is so small.


## 3. Specificity is combinatorial, and the kinetics cancel

A joint at length `L` can do three things, and the rates add up to

```
kext p + kext (1 - p) + koff(L) = kext + koff(L)
```

which does not contain `p`. The time spent at a given length is the same
whatever the sequence; only the branch taken differs. Writing
`s = kext / (kext + koff)` for the chance of surviving one step,

```
P_commit = Q * Prob[ Binomial(n, 1 - p) <= m ],   Q = s^n,   n = L_commit - k_seed
```

`Q` is the only place the rates enter, and `Q` is the same on both tracks.
So for any two donors, and for any tolerance `m`, the ratio
`P_commit(p1) / P_commit(p2)` contains no rate at all.

Two consequences. First, no stabilisation law can contribute to
specificity, because it acts identically on the donor and on the
background. Specificity comes from the number of consecutive checks
demanded and from nothing else. At the default settings, `n = 22`,
`Q = 0.950`, and the discrimination is `(1/0.271)^22 = 3.0e12`. Applied to
SHERPA, this says that its `koff1_adj` ladder and its `koff1_clip_nt`
floor are not the discriminating machinery, whatever they do to the
occupancy curves.

Second, the transit time conditional on success is
`sum_L 1/(kext + koff(L))`, which is also free of `p`; its full density is
Erlang with `n` steps for a flat `koff` (`theory.transit_time_pdf`). A
divergent donor is not slower to zip, only less likely to finish.
Divergence should move the yield of recombination without moving the
timing of the events that do succeed, which is a separable signature worth
looking for.


## 4. The divergence law, and what a tolerance does to it

With `p = 1 - delta` and no tolerance,

```
P_commit(delta) = (1 - delta)^n Q  ~  Q exp(-n delta)
```

so the slope of `log(yield)` against divergence is `n = L_commit - k_seed`
exactly. The exponential decline of homeologous recombination is the
model's first-order behaviour, not evidence for a mismatch-repair activity
of a given strength, and the slope measures the length of the test.
SHERPA's two candidate thresholds, `dloop_min_size = 18` and
`joint_l_min = 80`, predict slopes of 10 and 72, which a divergence series
tells apart. Fitting the isogenic curve cannot.

With `m > 0` the law acquires a shoulder. `Prob[Binomial(n, delta) <= m]`
is flat while `n delta` is small compared with `m` and recovers
`exp(-n delta)` beyond it, so the width of the shoulder measures `m` and
the slope past it still measures `n` (figure, panel A). This is the
natural image of SHERPA's mismatch tolerance, and it costs one integer
rather than a rate.

Fujitani et al. read the MEPS differently, as `2/sqrt(h)`, the length at
which a randomly walking branch point stops feeling the ends of the
homology, with an `N^3` law below it. The two readings answer different
experiments. Theirs is a length series on a fully homologous tract, which
is post-synaptic branch migration. This one is a divergence series at
fixed length, which is the presynaptic test. `L_commit` here is a
threshold on checks rather than on bases, and that is what makes it
measurable by divergence.


## 5. A fidelity bound derives the commitment length

Nothing so far says why a commitment threshold has to exist. This does.
Background seeds outnumber donor seeds `(1 - f) / f` to one, and each is
`(p_het/p_hom)^n` less likely to get through, so the odds of a false
commitment are `((1 - f)/f) (p_het/p_hom)^n`. Requiring them below a
tolerance `eps` gives a lower bound on the length of the test that
contains no kinetics and nothing fitted:

```
n >= ln( (1 - f) / (f eps) ) / ln( p_hom / p_het )
```

With the measured `f = 1.57e-3` and `p_het = 0.271`, on an isogenic donor:

| tolerated false rate | `n_min` | `L_commit >= k + n_min` |
| --- | --- | --- |
| 1e-2 | 8.5 | 17 |
| 1e-3 | 10.2 | 19 |
| 1e-4 | 12.0 | 20 |

SHERPA's `dloop_min_size = 18` was fitted. Here it is bracketed by a bound
with no free parameter in it. Read the other way, a threshold of 18 nt
admits 1.4e-3 false D-loops per true one, about one per thousand, which is
what one would have picked by hand. A fitted parameter landing on an
independently derived bound is the strongest single piece of evidence that
the mechanism SHERPA implements is the one described here.

The bound also shows the shape of the problem. `n_min` grows only
logarithmically in the required fidelity, so three more orders of
magnitude of specificity cost about five nucleotides. Recognition is cheap
once there is a per-nucleotide check. What it cannot buy cheaply is speed,
which is what the rest of SHERPA's machinery is for: Hi-C proximity,
filament mobility, resection timing.

Panel C draws the trade-off. On an isogenic donor the search time barely
moves as `L_commit` grows while the false-commitment odds fall by
twenty-seven decades. On a donor 5 % divergent the search time doubles
every fourteen nucleotides or so, and the threshold becomes a real design
variable. This is Bitran's speed against stringency, with their delay
`T_D` replaced by a length.

Savir and Tlusty make the same point in energy: the per-triplet free
energy of RecA is tuned so that the detection cost is near its minimum.
Here the per-step cost is `ln(p_hom/p_het) = 1.3` nats, set by the
alphabet and the genome composition rather than by the protein, and the
tuning variable is how many steps are demanded.


## 6. The search is an exact first-passage problem

Kochugaeva et al. solve their model through backward master equations and
Laplace transforms. The zipper allows something simpler. Its `N` sites are
independent, so

```
T_search = min(T_1, ..., T_N),   F_N(t) = 1 - (1 - F_1(t))^N
```

and `F_1`, the absorption of one site into the committed state starting
from free, is a phase-type distribution on the 91-state chain whose
generator is built from the very reaction list that produces the Antimony
source. On a uniform grid it costs one matrix exponential and one
vector-matrix product per time point, with no ensemble and no
approximation (`theory.search_time_distribution`).

What it shows, in panel D:

- the single-site hazard rises from zero over one transit time, about four
  seconds, and is flat after that. Once it is flat, `N h_1` equals
  `kon f S_free P_commit` to two parts in a thousand, so the steady-state
  rate used in sections 3 to 5 is the exact asymptotic hazard;
- the exact mean search time is 563.55 min against a steady-state estimate
  of 563.57 min;
- the Gillespie ensemble sits on the exact distribution. For 10 000 cells
  the Kolmogorov distance between the empirical and exact distribution
  functions is 0.006, against a 5 % critical value of 0.014; 57.5 ± 0.5 %
  of cells have committed by 480 min against 57.3 % exact; and the mean
  time of those that have is 206.5 min against 206.4 min.

This is also the clearest statement of what the model cannot produce. It
predicts an exponential first-commitment time after a transient of a few
seconds. Wiktor et al. measure, in *E. coli*, a gamma-shaped search time
with a mean of 9 min and a standard deviation of 3 min, which is strongly
peaked. No memoryless mechanism gives that shape. A peak has to come
either from steps before the test, such as resection and filament
assembly, which are a pure time offset and can be convolved in, or from a
search that accumulates progress: filament repositioning, reduced
dimensionality, local enrichment by chromosome conformation. Those are
exactly what SHERPA keeps and this model drops, so the shape of SHERPA's
first-commitment distribution after its resection delay is the experiment
that separates the two.

**Where the time goes.** Bitran et al. write the mean search time as
`<T> = (1/P_T) [tau_off + tau_diff + tau_target]`, the number of donor
encounters needed times the cost of one round. Little's law on the zipper
gives that identity exactly:

```
1 / commit_rate = (1 / P_commit) [ 1/(kon f) + ((1-f)/f) tau_het + tau_hom ] / N
```

with `tau_diff = 1/(kon f)` the wait for a donor encounter, `tau_off` the
time lost on the background between two of them, and `tau_target` the time
spent on the donor itself. At this calibration the three terms are 1.1e5,
9.1e2 and 0.07 min. The search is limited by nucleation, not by
off-targets, by two orders of magnitude. That is why `koff0` hardly
matters here, and why Bitran's kinetic-trapping regime, where `tau_off`
dominates, is the regime SHERPA would have to be in for its stabilisation
laws to matter to the search.


## 7. Clogging, and the best filament stability

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

a geometric mean of the nucleation and zipping rates. SHERPA's
`r_off1 = 0.7` sits a factor three above it, at 98 % of the attainable
rate.

That agreement should be reported carefully. At this nucleation load the
optimum is broad, with 1.3 decades of `koff0` staying within 5 % of the
maximum (panel B). The coincidence is not evidence that `r_off1` was tuned
by selection for search speed. It is evidence that in the dilute regime
the search is insensitive to filament stability altogether. Push the load
up by a factor 300 and the optimum sharpens, `S_free` falls to 56 out of
200, and `r_off1` would deliver half the attainable rate. Filament
stability becomes a design variable only once the filament is crowded,
which is the regime Kochugaeva et al. describe through their optimal
affinity `K = kon/koff` of order 1 to 10.


## 8. Does a constant `p` survive a real genome?

`p = 1/4` is the random-sequence null, and a real genome is not random.
`homozip spectrum` measures the real thing: for every 8-mer of the LY
filament, every exact match in S288c extended to its first mismatch. That
is 1 295 385 (site, locus) pairs and an empirical survival curve. Run it
twice, once with the LYS2 donor masked, and the two effects separate
cleanly. The measurements are kept in `resources/spectrum/` and drawn in
`spectrum.png`.

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

**The rise of the whole-genome `p(L)` to 1.0 is not a missing ingredient.**
It is the two-track structure seen from outside. Loci that have already
matched 13 nt are overwhelmingly the donor, because the background has
been thinned by `0.27^5`, so the surviving population stops being random.
That is a mixture effect, and `S(L) = (1 - f) p_het^(L-k) + f` reproduces
the measured curve with nothing added. The crossover at L ≈ 13 sits right
where the fidelity bound of section 5 put the threshold. This is also the
answer to the objection that `p` ought to grow as the match gets longer:
it does, in the aggregate, and the two-track model already predicts it.

**The genuine correction is the repeat tail.** Above L ≈ 15 the
donor-masked hazard rises to about 0.35 because of Ty elements,
paralogues and subtelomeric repeats. That makes the background about
thirty times more permissive at L = 30 than the flat law, against a
discrimination budget of 1e12, so the conclusions of sections 3 to 7 do
not move. The generator accepts the measured curve directly, and the exact
first passage works unchanged on it:

```python
from homozip import genome
from homozip.model import Params
bg = genome.load("resources/spectrum/background.json")
prm = Params(p_het=genome.p_profile(bg, 8, 30))
```

Bitran et al. weight their specificity by the genome's spectrum of
accidental matches. This is the same object, read the same way. One
caveat: above L ≈ 16 the masked spectrum rests on a few tens of pairs from
a single 2 kb filament, so the tail is not well determined. Measuring it
properly needs many filaments, or the genome against itself.


## 9. Three predictions for SHERPA

Each is cheap to test by re-running SHERPA, and each can fail.

1. **Flatten the ladder**, `koff1_adj = 1`. The zipper puts the whole
   ladder at 3.2 % of the commit rate. If SHERPA's homologous yield moves
   by much more than that, the ladder is acting on the D-loop lifetime
   rather than on the search.
2. **Run a divergence series.** `log(yield)` against `delta` should be a
   straight line of slope `-(L_commit - k_seed)`, which is 10 or 72
   depending on which threshold is rate-limiting, with a shoulder at low
   divergence if mismatches are tolerated. The timing of the successful
   events should not move at all.
3. **Look at the shape of the first commitment time**, after the resection
   delay. The zipper gives an exponential after a transient of a few
   seconds. A peak would mean the search accumulates progress, and would
   name what SHERPA adds.


## 10. What is left out, and what it costs

| left out | consequence | why it is acceptable here |
| --- | --- | --- |
| Hi-C weighting | `f` is a scalar, no 4C profile | the rate is linear in `f`, so a structured `f` reweights donors without touching the mechanism |
| filament mobility | `f` is constant in time | the search is memoryless, so under fast switching only the time-averaged `f` enters |
| Rad54, roadblocks, D-loop geometry | no D-loop size histogram | downstream of commitment, and identical on both tracks |
| genome and k-mer index | no per-locus prediction | replaced by `p` and `f`, the only two things the mechanism reads |
| Gamma resection delay | the search starts at t = 0 | a pure time offset; convolve the exact density with the Gamma to compare |
| second proofreading stage | one gate instead of two | two gates multiply, so a divergence slope alone cannot separate them |
| deterministic drift for `r_elong` | Poisson zipping | the mean is matched, the variance is not, which is harmless while `kext` is far above `koff0` |


## 11. SHERPA and the zipper, parameter by parameter

| SHERPA | zipper | relation |
| --- | --- | --- |
| `ksize` | `k_seed` | identical |
| `r_on` | `kon N` | `kon = r_on / N` |
| `r_elong` | `kext` | identical, and it sets the time unit |
| `r_off1` | `koff0` | identical |
| `koff1_adj`, per triplet | `lam`, off by default | `lam = 3 / ln(1/koff1_adj) = 8.41` nt, worth 3.2 % |
| `koff1_clip_nt` | dropped | not identifiable once `Q` is close to 1 |
| `mismatch_gap` | `max_mismatches` | the whole mismatch layer |
| `dloop_min_size`, `joint_l_min`, `joint_hill_n` | `L_commit` | merged into one threshold |
| `r_dloop`, `r_off2`, `r_dext` | dropped | downstream of commitment, non-discriminating |
| `hic_epsilon`, viewpoint, `r_jump*`, `tau_open` | `f` | collapsed to a scalar |
| `reject_stringency_S` | dropped | a second proofreading stage, see section 10 |


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
