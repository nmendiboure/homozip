"""
Antimony / SBML model generator for the Gillespie simulation of
RAD51-mediated strand exchange after a double-strand break.

Coarse-grained version of genomatch (the Monte-Carlo parent project):
no Hi-C weighting, no spatial geometry on the filament. Length
dependence is preserved through three rate-law families:

    - dissociation: continuous exponential decay      (k_off1)
    - D-loop formation: thresholded sigmoid           (p_dloop)
    - D-loop disruption: inverse-length scaling       (k_off2)

D-loops are size-resolved (DHM_L, DHT_L) so that k_off2(L) keeps its
length dependence, exactly as in genomatch.

Implementation note
-------------------
Tellurium's Antimony parser (this version) does not support
`piecewise` inside top-level `function` blocks. To stay consistent
with SBML semantics while keeping the source clean, the three rate
laws are inlined per-reaction using `piecewise(...)` directly in the
kinetic law. The exported SBML uses standard MathML <piecewise>.
"""

import hashlib
import json

import tellurium as te


# =====================================================================
# Helpers
# =====================================================================

def _stable_uid(config: dict) -> int:
    """SHA-256 based 32-bit UID, stable under YAML key reordering."""
    canon = json.dumps(config, sort_keys=True, default=str)
    return int(hashlib.sha256(canon.encode()).hexdigest(), 16) % 2**32


def _validate(intermediates: list[int]) -> None:
    if len(intermediates) < 2:
        raise ValueError("intermediates must contain at least 2 values.")
    if intermediates[0] < 1:
        raise ValueError("First bucket must have length >= 1 nt.")
    if any(b <= a for a, b in zip(intermediates, intermediates[1:])):
        raise ValueError("intermediates must be strictly increasing.")


# =====================================================================
# Inline rate-law fragments (Antimony / MathML)
# =====================================================================
# Each helper returns the *expression* string (without trailing
# `* species`) that encodes the corresponding law. The numeric L is
# baked in at generation time; everything else stays symbolic so the
# user can still tune koff1, koff1_alpha, ... in the YAML and rebuild.

def _expr_k_off1(L: int) -> str:
    """k_off1(L) = max( koff1 * exp(-koff1_alpha * (L - koff1_lref)),  koff1_floor )"""
    decay = f"koff1 * exp(-koff1_alpha * ({L} - koff1_lref))"
    return f"piecewise({decay}, {decay} > koff1_floor, koff1_floor)"


def _expr_k_off2(L: int) -> str:
    """k_off2(L) = koff2  if L <= koff2_lref  else  koff2 * koff2_lref / L"""
    return f"piecewise(koff2, {L} <= koff2_lref, koff2 * koff2_lref / {L})"


def _expr_p_dloop(L: int) -> str:
    """p_dloop(L) = 0  if L < dloop_lmin  else  sigmoid((L - L_half) / w)"""
    sig = f"1 / (1 + exp(-({L} - dloop_l_half) / dloop_w))"
    return f"piecewise(0, {L} < dloop_lmin, {sig})"


# =====================================================================
# Main entry point
# =====================================================================

def generate_gillespie_model(config: dict) -> tuple[str, dict, int]:
    """
    Build an Antimony model string from a YAML-loaded config.

    Returns
    -------
    antimony_str : str
        Source of the model, ready for `tellurium.loada`.
    index_to_species : dict[int, str]
        Mapping output-row index -> species name. Index 0 is "Time",
        index 1 is "S", and the remaining indices follow the species
        declaration order in the model.
    uid : int
        Stable 32-bit identifier derived from the config.
    """

    uid = _stable_uid(config)

    # =================================================================
    # Pull the config
    # =================================================================
    N            = config["N"]
    f            = config["f"]
    intermediates: list[int] = list(config["intermediates"])
    _validate(intermediates)

    kon          = config["kon"]
    koff1        = config["koff1"]
    koff1_alpha  = config["koff1_alpha"]
    koff1_floor  = config["koff1_floor"]
    koff1_lref   = config["koff1_lref"]

    kext         = config["kext"]
    eps_mm       = config["eps_mm"]

    kdloop       = config["kdloop"]
    dloop_lmin   = config["dloop_lmin"]
    dloop_l_half = config["dloop_l_half"]
    dloop_w      = config["dloop_w"]

    koff2        = config["koff2"]
    koff2_lref   = config["koff2_lref"]

    kre          = config["kre"]

    L0 = intermediates[0]

    # =================================================================
    # Build species index in declaration order
    # =================================================================
    # Tellurium emits simulation columns in declaration order, with
    # Time as the implicit column 0. Keep these two lists in sync.
    index_to_species: dict[int, str] = {0: "Time", 1: "S"}
    declared: list[str] = ["S"]

    for prefix in ("HM", "HT", "DHM", "DHT"):
        for L in intermediates:
            declared.append(f"{prefix}{L}")
    declared.append("R")

    for i, name in enumerate(declared, start=1):
        index_to_species[i] = name

    # =================================================================
    # Assemble the Antimony source
    # =================================================================
    out: list[str] = []

    out.append("model rad51_recombination()\n")
    out.append("    compartment cell = 1;\n\n")

    # ---- Parameters ----
    out.append("    // ---- Parameters ----\n")
    for name, value in (
        ("N",            N),
        ("f",            f),
        ("kon",          kon),
        ("koff1",        koff1),
        ("koff1_alpha",  koff1_alpha),
        ("koff1_floor",  koff1_floor),
        ("koff1_lref",   koff1_lref),
        ("kext",         kext),
        ("eps_mm",       eps_mm),
        ("kdloop",       kdloop),
        ("dloop_lmin",   dloop_lmin),
        ("dloop_l_half", dloop_l_half),
        ("dloop_w",      dloop_w),
        ("koff2",        koff2),
        ("koff2_lref",   koff2_lref),
        ("kre",          kre),
    ):
        out.append(f"    {name} = {value};\n")
    out.append("\n")

    # ---- Species declarations ----
    out.append("    // ---- Species ----\n")
    out.append("    species S in cell;\n")
    out.append("    // Pre-synaptic complexes (homologous / heterologous)\n")
    for prefix in ("HM", "HT"):
        names = ", ".join(f"{prefix}{L}" for L in intermediates)
        out.append(f"    species {names} in cell;\n")
    out.append("    // Size-resolved D-loops (homologous / heterologous)\n")
    for prefix in ("DHM", "DHT"):
        names = ", ".join(f"{prefix}{L}" for L in intermediates)
        out.append(f"    species {names} in cell;\n")
    out.append("    species R in cell;\n\n")

    # ---- Initial state ----
    out.append("    // ---- Initial state ----\n")
    out.append(f"    S = {N};\n")
    for prefix in ("HM", "HT", "DHM", "DHT"):
        for L in intermediates:
            out.append(f"    {prefix}{L} = 0;\n")
    out.append("    R = 0;\n\n")

    # =================================================================
    # Reactions
    # =================================================================
    rid = 1

    def emit(line: str) -> None:
        nonlocal rid
        out.append(f"    R{rid}: {line}\n")
        rid += 1

    # ---- Association ----
    out.append("    // ---- Association ----\n")
    emit(f"S -> HM{L0}; kon * f * S;")
    emit(f"S -> HT{L0}; kon * (1 - f) * S;")

    # ---- Dissociation: same continuous law for HM and HT ----
    for label in ("HM", "HT"):
        out.append(f"\n    // ---- Dissociation ({label}) ----\n")
        for L in intermediates:
            emit(f"{label}{L} -> S; ({_expr_k_off1(L)}) * {label}{L};")

    # ---- Extension: per-nt rate rescaled by bucket spacing ----
    # The per-bucket transition rate is kext / Δ, where Δ is the gap
    # between consecutive bucket lengths. This makes kext interpretable
    # as "nucleotides extended per unit time", independent of the
    # discretization choice. HT is gated by eps_mm.
    for label in ("HM", "HT"):
        suffix = " * eps_mm" if label == "HT" else ""
        out.append(f"\n    // ---- Extension ({label}) ----\n")
        for a, b in zip(intermediates, intermediates[1:]):
            delta = b - a
            emit(
                f"{label}{a} -> {label}{b}; "
                f"(kext / {delta}){suffix} * {label}{a};"
            )

    # ---- D-loop formation: HM_L -> DHM_L, HT_L -> DHT_L ----
    # The target keeps the length, so k_off2(L) and DLC histograms
    # remain well-defined downstream.
    for label in ("HM", "HT"):
        d_label = "DHM" if label == "HM" else "DHT"
        suffix = " * eps_mm" if label == "HT" else ""
        out.append(f"\n    // ---- D-loop formation ({label}) ----\n")
        for L in intermediates:
            emit(
                f"{label}{L} -> {d_label}{L}; "
                f"kdloop * ({_expr_p_dloop(L)}){suffix} * {label}{L};"
            )

    # ---- D-loop disruption: inverse-length scaling ----
    for d_label in ("DHM", "DHT"):
        out.append(f"\n    // ---- D-loop disruption ({d_label}) ----\n")
        for L in intermediates:
            emit(f"{d_label}{L} -> S; ({_expr_k_off2(L)}) * {d_label}{L};")

    # ---- Recombination: any D-loop can transition to R at rate kre ----
    out.append("\n    // ---- Recombination ----\n")
    for d_label in ("DHM", "DHT"):
        for L in intermediates:
            emit(f"{d_label}{L} -> R; kre * {d_label}{L};")

    out.append("end\n")

    return "".join(out), index_to_species, uid


# =====================================================================
# SBML export
# =====================================================================

def export_sbml(antimony_str: str) -> str:
    """Compile the Antimony source to an SBML L3v2 XML string."""
    return te.antimonyToSBML(antimony_str)
