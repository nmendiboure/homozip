"""homozip theory | figures | spectrum

Runnable as a script (python src/homozip/main.py ...) or through the
homozip entry point declared in pyproject.toml; the imports are absolute
so that both work.
"""

from __future__ import annotations

import argparse
import math
import os

import yaml

from homozip import __version__, model
from homozip.model import Params


def load_params(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as fh:
        return Params.from_config(yaml.safe_load(fh) or {})


def _fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


# =====================================================================
# Commands
# =====================================================================

def cmd_theory(args) -> None:
    prm = load_params(args.params)
    summary = model.summarize(prm, exact=not args.no_exact)
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"{key:<{width}}  {_fmt(value)}")

    if prm.max_mismatches == 0 and isinstance(prm.p_hom, float) \
            and isinstance(prm.p_het, float):
        print("\nfidelity bound:  n >= ln((1-f)/(f c eps)) / ln(p_hom/p_het)")
        for eps in (1e-2, 1e-3, 1e-4):
            n_min = model.min_commitment_steps(prm, eps)
            print(f"  eps = {eps:<7g} n_min = {n_min:5.1f}   "
                  f"L_commit >= {prm.k_seed + math.ceil(n_min)}")

    if prm.lam is None:
        # What a stabilisation ladder would be worth here, taking koff down by
        # a factor 0.7 per triplet as a plausible per-turn stabilisation.
        lam_ladder = 3.0 / math.log(1.0 / 0.7)
        gain = model.commit_rate(prm.with_(lam=lam_ladder)) / model.commit_rate(prm)
        print(f"\nkoff(L) ladder, lam = {lam_ladder:.2f} nt: commit rate x {gain:.4f} "
              f"({100 * (gain - 1):+.1f} %)")


def cmd_figures(args) -> None:
    from homozip import figures
    prm = load_params(args.params)

    print("closed forms:")
    for key, value in model.summarize(prm, exact=False).items():
        print(f"  {key:<22} {_fmt(value)}")

    paths = figures.figure_model(prm, args.output)
    if os.path.isfile(args.genome_spectrum) and os.path.isfile(args.background_spectrum):
        paths += figures.figure_spectrum(args.genome_spectrum,
                                         args.background_spectrum,
                                         args.output, prm.l_commit)
    for path in paths:
        print(f"figure   {path}")


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


# =====================================================================
# Parser
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="homozip",
        description="the mismatch-limited zipper, a minimal exactly solvable "
                    "model of homology search")
    ap.add_argument("--version", action="version", version=f"homozip {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("theory", help="print the closed-form summary")
    p.add_argument("params", nargs="?", default="params.yaml")
    p.add_argument("--no-exact", action="store_true",
                   help="skip the exact mean search time")
    p.set_defaults(func=cmd_theory)

    p = sub.add_parser("figures", help="one PDF per panel")
    p.add_argument("params", nargs="?", default="params.yaml")
    p.add_argument("-o", "--output", default="output/fig")
    p.add_argument("--genome-spectrum", default="resources/spectrum/genome.json")
    p.add_argument("--background-spectrum",
                   default="resources/spectrum/background.json")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("spectrum", help="measure the match-length spectrum")
    p.add_argument("--genome", default="resources/S288c-Lys2.fa", help="FASTA")
    p.add_argument("--filament", default="resources/LY.yaml",
                   help="filament YAML, `name: sequence` or `filament: {sequence}`")
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
