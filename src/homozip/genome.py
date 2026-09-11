"""
Match-length spectrum of a filament against a real genome.

This is where p_het and f come from. For every k-mer of the filament,
every exact match in the genome is extended to its first mismatch, which
gives the longest exact match M of each (filament site, genomic locus)
pair, that is, the table a nucleation draws from. From the distribution
of M:

    survival   P(M >= L)
    hazard     p(L) = P(M >= L+1) / P(M >= L)

p(L) is the model's p, measured rather than fitted. Mask the donor and the
flat part of p(L) gives p_het; keep it and the plateau of the survival
curve at L_commit gives f.
"""

from __future__ import annotations

import json

import numpy as np
import yaml


CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    CODE[_b] = _i
    CODE[_b + 32] = _i          # lower case too
COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


# =====================================================================
# I/O
# =====================================================================

def read_fasta(path: str) -> dict[str, np.ndarray]:
    """FASTA to {name: uint8 array}, A=0 C=1 G=2 T=3, anything else 4."""
    seqs: dict[str, list[bytes]] = {}
    name = None
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                name = line[1:].split()[0].decode()
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {k: CODE[np.frombuffer(b"".join(v), dtype=np.uint8)]
            for k, v in seqs.items()}


def read_filament(path: str) -> np.ndarray:
    """The ssDNA sequence of a SHERPA filament YAML."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    seq = data["filament"]["sequence"].strip()
    return CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]


def parse_mask(spec: str) -> tuple[str, int, int]:
    """'chr2:471858-473926' to ('chr2', 471858, 473926), 1-based inclusive."""
    name, span = spec.split(":")
    start, end = (int(v) for v in span.split("-"))
    return name, start, end


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save(spec: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)


# =====================================================================
# The genome as one searchable array, both strands
# =====================================================================

def build_genome_array(chroms: dict[str, np.ndarray], pad: int):
    """Every chromosome, then its reverse complement, into one array.
    Blocks are separated by invalid bases so that no k-mer and no extension
    can run across a junction."""
    blocks, ids, names = [], [], []
    sep = np.full(pad, 4, dtype=np.uint8)
    for strand, transform in (("+", lambda a: a),
                              ("-", lambda a: COMPLEMENT[a][::-1])):
        for chrom, arr in chroms.items():
            block = transform(arr)
            blocks.append(block)
            ids.append(np.full(block.size, len(names), dtype=np.int16))
            names.append(f"{chrom}{strand}")
            blocks.append(sep)
            ids.append(np.full(pad, -1, dtype=np.int16))
    return np.concatenate(blocks), np.concatenate(ids), names


def kmer_codes(seq: np.ndarray, ksize: int) -> np.ndarray:
    """Rolling k-mer code per start position, -1 where a base is invalid."""
    n = seq.size - ksize + 1
    codes = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    for j in range(ksize):
        col = seq[j:j + n]
        codes = codes * 4 + np.where(col < 4, col, 0)
        bad |= col >= 4
    codes[bad] = -1
    return codes


# =====================================================================
# The measurement
# =====================================================================

def match_lengths(filament: np.ndarray, genome: np.ndarray, block_id: np.ndarray,
                  ksize: int, max_extend: int):
    """Longest exact match of every (filament site, genomic locus) pair that
    shares a k-mer, plus the genomic block each pair came from."""
    g_codes = kmer_codes(genome, ksize)
    valid = np.flatnonzero(g_codes >= 0)
    order = valid[np.argsort(g_codes[valid], kind="stable")]
    sorted_codes = g_codes[order]

    f_codes = kmer_codes(filament, ksize)
    window = np.arange(ksize, ksize + max_extend)
    all_len, all_blk = [], []

    for i, code in enumerate(f_codes):
        if code < 0:
            continue
        lo, hi = np.searchsorted(sorted_codes, [code, code + 1])
        if hi <= lo:
            continue
        pos = order[lo:hi]

        # The filament window 3' of the seed, padded past its end with a
        # code that can never match.
        if i + ksize + max_extend <= filament.size:
            f_win = filament[i + window]
        else:
            pad = i + ksize + max_extend - filament.size
            f_win = np.concatenate([filament[i + ksize:],
                                    np.full(pad, 5, dtype=np.uint8)])

        g_idx = pos[:, None] + window[None, :]
        np.clip(g_idx, 0, genome.size - 1, out=g_idx)
        mism = (genome[g_idx] != f_win[None, :]) | (genome[g_idx] >= 4)
        first = np.where(mism.any(axis=1), mism.argmax(axis=1), max_extend)

        all_len.append((ksize + first).astype(np.int32))
        all_blk.append(block_id[pos])

    return np.concatenate(all_len), np.concatenate(all_blk)


def spectrum(lengths: np.ndarray, ksize: int, l_max: int) -> dict:
    grid = np.arange(ksize, l_max + 1)
    surv = np.array([(lengths >= L).mean() for L in grid])
    haz = np.divide(surv[1:], surv[:-1], out=np.zeros(grid.size - 1),
                    where=surv[:-1] > 0)
    return {"ksize": int(ksize), "L": grid.tolist(), "survival": surv.tolist(),
            "hazard_L": grid[:-1].tolist(), "hazard": haz.tolist(),
            "n_pairs": int(lengths.size)}


def measure(genome_fasta: str, filament_yaml: str, ksize: int = 8,
            max_extend: int = 56, l_max: int = 40, masks: tuple[str, ...] = (),
            log=print) -> dict:
    """The whole measurement. Masked intervals are blanked before indexing,
    which is how the donor is removed to see the background alone."""
    chroms = read_fasta(genome_fasta)
    for spec in masks:
        name, start, end = parse_mask(spec)
        chroms[name][start - 1:end] = 4
        log(f"masked      {name}:{start}-{end} ({end - start + 1} bp)")

    filament = read_filament(filament_yaml)
    log(f"filament    {filament.size} nt")
    log(f"genome      {sum(a.size for a in chroms.values()):,} bp, "
        f"{len(chroms)} sequences")

    genome, block_id, names = build_genome_array(chroms, pad=max_extend + ksize)
    lengths, blocks = match_lengths(filament, genome, block_id, ksize, max_extend)
    log(f"pairs       {lengths.size:,} (filament site, genomic locus)")

    spec = spectrum(lengths, ksize, l_max)
    spec["masks"] = list(masks)
    long_pairs = lengths >= 30
    spec["pairs_reaching_30"] = {}
    if long_pairs.any():
        uniq, cnt = np.unique(blocks[long_pairs], return_counts=True)
        spec["pairs_reaching_30"] = {names[b]: int(c) for b, c in zip(uniq, cnt)}
    return spec


# =====================================================================
# From a spectrum to model parameters
# =====================================================================

def background_p(spec: dict, l_from: int = 8, l_to: int = 14) -> float:
    """p_het, the mean hazard over the flat part of a donor-masked spectrum.
    On S288c the hazard is flat up to L = 14 and starts to feel the repeats
    at 15."""
    h = [v for l, v in zip(spec["hazard_L"], spec["hazard"])
         if l_from <= l <= l_to and v > 0]
    return float(np.mean(h))


def donor_fraction(spec: dict, l_commit: int) -> float:
    """f, the survival plateau of a whole-genome spectrum at L_commit."""
    return float(spec["survival"][spec["L"].index(l_commit)])


def p_profile(spec: dict, k_seed: int, l_commit: int) -> tuple[float, ...]:
    """A measured p(L) ready for Params.p_het. Lengths past the measured
    range reuse the last measured value."""
    table = {int(l): float(h) for l, h in zip(spec["hazard_L"], spec["hazard"])
             if h > 0.0}
    if not table:
        raise ValueError("empty hazard curve")
    last = table[max(table)]
    return tuple(table.get(l, last) for l in range(k_seed, l_commit))
