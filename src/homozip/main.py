"""homozip build | theory | run | spectrum | figure

Runnable as a script (python src/homozip/main.py ...) or through the
homozip entry point declared in pyproject.toml; the imports are absolute
so that both work.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import yaml

from homozip import __version__, theory
from homozip.model import Params, build_network, to_antimony, export_sbml, MODEL_NAME


def load_params(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as fh:
        return Params.from_config(yaml.safe_load(fh) or {})


def _fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


# =====================================================================
# Commands
# =====================================================================

def cmd_build(args) -> None:
    prm = load_params(args.params)
    net = build_network(prm)
    src = to_antimony(net, prm)

    outdir = os.path.join(args.output, "model")
    os.makedirs(outdir, exist_ok=True)
    txt = os.path.join(outdir, f"{MODEL_NAME}.txt")
    xml = os.path.join(outdir, f"{MODEL_NAME}.xml")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(xml, "w", encoding="utf-8") as fh:
        fh.write(export_sbml(src))

    print(f"uid        {prm.uid()}")
    print(f"species    {len(net.species)}")
    print(f"reactions  {len(net.reactions)}")
    print(f"antimony   {txt}")
    print(f"sbml       {xml}")


def cmd_theory(args) -> None:
    prm = load_params(args.params)
    summary = theory.summarize(prm, exact=not args.no_exact)
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"{key:<{width}}  {_fmt(value)}")

    if prm.max_mismatches == 0 and isinstance(prm.p_hom, float) \
            and isinstance(prm.p_het, float):
        print("\nfidelity bound:  n >= ln((1-f)/(f eps)) / ln(p_hom/p_het)")
        for eps in (1e-2, 1e-3, 1e-4):
            n_min = theory.min_commitment_steps(prm, eps)
            print(f"  eps = {eps:<7g} n_min = {n_min:5.1f}   "
                  f"L_commit >= {prm.k_seed + math.ceil(n_min)}")

    if prm.lam is None:
        # What SHERPA's per-triplet koff1_adj = 0.7 would be worth here.
        lam_sherpa = 3.0 / math.log(1.0 / 0.7)
        gain = theory.commit_rate(prm.with_(lam=lam_sherpa)) / theory.commit_rate(prm)
        print(f"\nkoff(L) ladder, lam = {lam_sherpa:.2f} nt: commit rate x {gain:.4f} "
              f"({100 * (gain - 1):+.1f} %)")


def cmd_run(args) -> None:
    from homozip import simulate
    prm = load_params(args.params)
    n_cells = args.cells or prm.n_cells
    print(f"running {n_cells} cells, t_end = {prm.t_end} min, uid {prm.uid()}",
          flush=True)

    ens = simulate.run_ensemble(prm, n_cells, progress=True)
    t, times = ens["t"], ens["first_commit"]
    committed = times[~np.isnan(times)]
    dist = theory.search_time_distribution(prm, t)
    ks = float(np.max(np.abs(simulate.empirical_cdf(times, t) - dist["cdf"])))
    frac = committed.size / times.size
    se = math.sqrt(max(frac * (1 - frac), 1e-12) / times.size)

    outdir = os.path.join(args.output, "run")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"ensemble_{prm.uid()}.npz")
    np.savez_compressed(path, **ens, exact_cdf=dist["cdf"], exact_pdf=dist["pdf"])

    print(f"\ncommitted by t_end   SSA {frac:.3f} +/- {se:.3f}   "
          f"exact {dist['cdf'][-1]:.3f}")
    if committed.size:
        conditional = float(np.trapz(t * dist["pdf"], t) / dist["cdf"][-1])
        print(f"mean of committed    SSA {committed.mean():.1f} min   "
              f"exact {conditional:.1f} min")
    print(f"unconditional mean   exact {theory.mean_search_time_exact(prm):.1f} min   "
          f"steady state {theory.mean_search_time_qss(prm):.1f} min")
    print(f"KS distance          {ks:.4f}")
    print(f"free sites           SSA {ens['S_mean'].mean():.2f}   "
          f"steady state {theory.free_sites(prm):.2f}")
    print(f"written {path}")


def cmd_spectrum(args) -> None:
    from homozip import genome
    spec = genome.measure(args.genome, args.filament, ksize=args.ksize,
                          max_extend=args.max_extend, l_max=args.l_max,
                          masks=tuple(args.mask))

    print(f"\n{'L':>4} {'P(M>=L)':>12} {'null':>12} {'ratio':>9} {'p(L)':>8}")
    for j, L in enumerate(spec["L"]):
        if L > args.ksize and (L <= args.ksize + 8 or L % 4 == 0):
            null = 0.25 ** (L - args.ksize)
            print(f"{L:>4} {spec['survival'][j]:>12.3e} {null:>12.3e} "
                  f"{spec['survival'][j] / null:>9.3g} {spec['hazard'][j - 1]:>8.3f}")

    if spec["pairs_reaching_30"]:
        print("\npairs reaching L >= 30, by block:")
        for name, count in sorted(spec["pairs_reaching_30"].items(),
                                  key=lambda kv: -kv[1])[:8]:
            print(f"   {name:>8}  {count:,}")

    print(f"\nmean hazard over L = 8..14 (this is p_het when the donor is masked)"
          f"  {genome.background_p(spec):.4f}")
    if args.l_commit <= args.l_max:
        print(f"survival at L = {args.l_commit} (this is f when it is not)"
              f"  {genome.donor_fraction(spec, args.l_commit):.3e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    genome.save(spec, args.output)
    print(f"written {args.output}")


def cmd_figure(args) -> None:
    from homozip import figures
    prm = load_params(args.params)
    fig_dir = os.path.join(args.output, "fig")

    print("closed forms:")
    for key, value in theory.summarize(prm, exact=False).items():
        print(f"  {key:<22} {_fmt(value)}")

    print(f"figure   {figures.figure_model(prm, fig_dir, quick=args.quick)}")
    if os.path.isfile(args.genome_spectrum) and os.path.isfile(args.background_spectrum):
        out = figures.figure_spectrum(args.genome_spectrum, args.background_spectrum,
                                      fig_dir, prm.l_commit)
        print(f"figure   {out}")


# =====================================================================
# Parser
# =====================================================================

def _model_command(sub, name: str, func, help: str, output: bool = True):
    """A subcommand that reads params.yaml and, usually, writes under -o."""
    p = sub.add_parser(name, help=help)
    p.add_argument("params", nargs="?", default="params.yaml")
    if output:
        p.add_argument("-o", "--output", default="output")
    p.set_defaults(func=func)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="homozip",
        description="the mismatch-limited zipper, a minimal stochastic model "
                    "of homology search")
    ap.add_argument("--version", action="version", version=f"homozip {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    _model_command(sub, "build", cmd_build, "write the Antimony and SBML model")

    p = _model_command(sub, "theory", cmd_theory, "print the closed-form summary",
                       output=False)
    p.add_argument("--no-exact", action="store_true",
                   help="skip the exact mean search time")

    p = _model_command(sub, "run", cmd_run,
                       "Gillespie ensemble against the exact solution")
    p.add_argument("-c", "--cells", type=int, default=None)

    p = _model_command(sub, "figure", cmd_figure, "the figures")
    p.add_argument("--quick", action="store_true", help="small ensembles")
    p.add_argument("--genome-spectrum", default="resources/spectrum/genome.json")
    p.add_argument("--background-spectrum",
                   default="resources/spectrum/background.json")

    p = sub.add_parser("spectrum", help="measure the match-length spectrum")
    p.add_argument("--genome", default="resources/S288c-Lys2.fa", help="FASTA")
    p.add_argument("--filament", default="resources/LY.yaml",
                   help="filament YAML, `name: sequence` or SHERPA's `filament: {sequence}`")
    p.add_argument("--ksize", type=int, default=8)
    p.add_argument("--max-extend", type=int, default=56)
    p.add_argument("--l-max", type=int, default=40)
    p.add_argument("--l-commit", type=int, default=30)
    p.add_argument("--mask", action="append", default=[], metavar="CHR:START-END",
                   help="blank a genomic interval before indexing (repeatable)")
    p.add_argument("-o", "--output", default="output/spectrum.json")
    p.set_defaults(func=cmd_spectrum)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
