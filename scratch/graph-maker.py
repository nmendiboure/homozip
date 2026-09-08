"""
Gillespie model — horizontal flow diagram (post-refactor v2).

Renders the *current* Antimony model produced by
`sherpassa.modelmaker.generate_gillespie_model`, with:

  - explicit length-resolved D-loop states  (DHM_L, DHT_L)
  - continuous rate laws  k_off1, k_off2, p_dloop
  - per-nt extension rate scaled by bucket spacing
  - eps_mm gating extension AND D-loop formation on the HT branch

The graph is laid out left-to-right and split in horizontal lanes so
that the homologous and heterologous tracks read in parallel. Suitable
for slide insertion (PNG / SVG / PDF).

Outputs (next to this script):
    gillespie_model_v2.svg
    gillespie_model_v2.png
    gillespie_model_v2.pdf
"""

from graphviz import Digraph


# =====================================================================
# Canvas
# =====================================================================
dot = Digraph(format="svg", engine="dot")
dot.attr(rankdir="LR", splines="spline", overlap="false",
         nodesep="0.5", ranksep="1.4", size="16,9!", ratio="fill")
dot.attr("graph", fontname="Helvetica", fontsize="14", labelloc="t",
         label="RAD51 strand-exchange Gillespie model — refactor v2")
dot.attr("node",  fontname="Helvetica", shape="box", style="filled,rounded",
         fontsize="12", penwidth="1.4", margin="0.18,0.12")
dot.attr("edge",  fontname="Helvetica", fontsize="11", arrowsize="0.85")


# =====================================================================
# Color palette
# =====================================================================
COL_S        = "#dcdcdc"   # free filament
COL_HM       = "#bcd5f0"   # homologous track
COL_HM_DARK  = "#5b8fc7"
COL_HT       = "#f3c0bc"   # heterologous track
COL_HT_DARK  = "#c47266"
COL_DLOOP_H  = "#8fb8e8"
COL_DLOOP_T  = "#e89c93"
COL_RECOMB   = "#ffd966"
COL_RETURN   = "#ffffff"
COL_LEGEND   = "#fafafa"


# =====================================================================
# 1) Source / sink
# =====================================================================
with dot.subgraph(name="cluster_source") as g:
    g.attr(label="", style="invis")
    g.node("S",
           "S\nFree microhomology slots\n(N = 200, f = 0.002)",
           fillcolor=COL_S)
    g.node("R",
           "R\nRecombined\n(absorbing state)",
           fillcolor=COL_RECOMB, shape="box")


# =====================================================================
# 2) Homologous track (top lane)
# =====================================================================
with dot.subgraph(name="cluster_hm") as g:
    g.attr(label="Homologous track   (fraction f of associations)",
           style="rounded", color=COL_HM_DARK, fontcolor=COL_HM_DARK,
           bgcolor="#f3f8fd")

    g.node("HM_chain",
           "HM_L  pre-synaptic complex\n"
           "L ∈ {8, 9, 12, 15, 21, 24, 30, 45, 65, 95, 140, 200}\n"
           "12 length buckets",
           fillcolor=COL_HM)
    g.node("DHM_chain",
           "DHM_L  D-loop (size-resolved)\n"
           "one species per bucket L\n"
           "preserves length for k_off2(L)",
           fillcolor=COL_DLOOP_H)


# =====================================================================
# 3) Heterologous track (bottom lane)
# =====================================================================
with dot.subgraph(name="cluster_ht") as g:
    g.attr(label="Heterologous track   (fraction 1-f of associations)",
           style="rounded", color=COL_HT_DARK, fontcolor=COL_HT_DARK,
           bgcolor="#fdf3f1")

    g.node("HT_chain",
           "HT_L  pre-synaptic complex\n"
           "same buckets as HM",
           fillcolor=COL_HT)
    g.node("DHT_chain",
           "DHT_L  D-loop (size-resolved)",
           fillcolor=COL_DLOOP_T)


# =====================================================================
# 4) Forward edges — association
# =====================================================================
dot.edge("S", "HM_chain",
         label="kon · f · S\n→ HM_8",
         color=COL_HM_DARK, fontcolor=COL_HM_DARK)
dot.edge("S", "HT_chain",
         label="kon · (1 - f) · S\n→ HT_8",
         color=COL_HT_DARK, fontcolor=COL_HT_DARK)


# =====================================================================
# 5) Forward edges — extension (looping on the chain)
# =====================================================================
dot.edge("HM_chain", "HM_chain",
         label="(kext / Δ) · HM_L\nbucket-to-bucket\nΔ = L_{i+1} - L_i",
         color=COL_HM_DARK, fontcolor=COL_HM_DARK)
dot.edge("HT_chain", "HT_chain",
         label="(kext / Δ) · eps_mm · HT_L\nmismatch-gated extension",
         color=COL_HT_DARK, fontcolor=COL_HT_DARK)


# =====================================================================
# 6) Forward edges — D-loop formation
# =====================================================================
dot.edge("HM_chain", "DHM_chain",
         label="kdloop · p_dloop(L) · HM_L\n"
               "p_dloop = 0  if L < dloop_lmin\n"
               "       = σ((L - L½)/w)  otherwise",
         color=COL_HM_DARK, fontcolor=COL_HM_DARK)
dot.edge("HT_chain", "DHT_chain",
         label="kdloop · p_dloop(L) · eps_mm · HT_L\n"
               "(same sigmoid, gated by eps_mm)",
         color=COL_HT_DARK, fontcolor=COL_HT_DARK)


# =====================================================================
# 7) Forward edges — recombination
# =====================================================================
dot.edge("DHM_chain", "R", label="kre · DHM_L", color=COL_HM_DARK,
         fontcolor=COL_HM_DARK)
dot.edge("DHT_chain", "R", label="kre · DHT_L", color=COL_HT_DARK,
         fontcolor=COL_HT_DARK)


# =====================================================================
# 8) Reverse edges — dissociation back to S (dashed)
# =====================================================================
# Pre-synaptic complex disruption (k_off1, length-dependent)
dot.edge("HM_chain", "S",
         label="k_off1(L) · HM_L\nk_off1 = max(koff1·exp(-α(L-Lref)), floor)",
         style="dashed", constraint="false",
         color=COL_HM_DARK, fontcolor=COL_HM_DARK)
dot.edge("HT_chain", "S",
         label="k_off1(L) · HT_L\n(same law)",
         style="dashed", constraint="false",
         color=COL_HT_DARK, fontcolor=COL_HT_DARK)

# D-loop disruption (k_off2, inverse-length stabilization, genomatch-faithful)
dot.edge("DHM_chain", "S",
         label="k_off2(L) · DHM_L\nk_off2 = koff2 · min(1, koff2_lref / L)",
         style="dashed", constraint="false",
         color=COL_HM_DARK, fontcolor=COL_HM_DARK)
dot.edge("DHT_chain", "S",
         label="k_off2(L) · DHT_L",
         style="dashed", constraint="false",
         color=COL_HT_DARK, fontcolor=COL_HT_DARK)


# Legend is rendered as a separate diagram (see below) so it does not
# compete with the flow diagram for vertical real estate.


# =====================================================================
# Render flow diagram
# =====================================================================
out_base = "gillespie_model_v2"
for fmt in ("svg", "png", "pdf"):
    dot.render(out_base, cleanup=True, format=fmt)
print(f"Wrote {out_base}.svg/.png/.pdf")


# =====================================================================
# Companion legend slide (parameters + rate laws), separate canvas
# =====================================================================
leg = Digraph(format="svg", engine="dot")
leg.attr(rankdir="TB", size="14,8!", ratio="fill")
leg.attr("graph", fontname="Helvetica", fontsize="14", labelloc="t",
         label="RAD51 Gillespie model — parameters and rate laws")
leg.attr("node", fontname="Helvetica", shape="note",
         fontsize="12", penwidth="1.0", style="filled", fillcolor="white")

params_text = (
    "Parameters\\l"
    "─────────────────────────────\\l"
    "N             population of microhomology slots\\l"
    "f             homologous donor fraction\\l"
    "kon           association rate\\l"
    "koff1         base dissociation rate (at L = koff1_lref)\\l"
    "koff1_alpha   exp. decay slope of k_off1 (per nt)\\l"
    "koff1_floor   lower clamp on k_off1\\l"
    "koff1_lref    reference length for k_off1\\l"
    "kext          extension rate (per nt)\\l"
    "eps_mm        mismatch tolerance (HT only)\\l"
    "kdloop        D-loop formation base rate\\l"
    "dloop_lmin    hard threshold L < lmin → p_dloop = 0\\l"
    "dloop_l_half  sigmoid midpoint\\l"
    "dloop_w       sigmoid width\\l"
    "koff2         base D-loop disruption rate\\l"
    "koff2_lref    D-loop length above which it stabilizes\\l"
    "kre           recombination rate\\l"
    "intermediates length buckets (12 values)\\l"
)
leg.node("params", params_text)

laws_text = (
    "Rate laws\\l"
    "─────────────────────────────\\l"
    "k_off1(L) = max( koff1 · exp(-koff1_alpha·(L - koff1_lref)),  koff1_floor )\\l"
    "\\l"
    "k_off2(L) = koff2                            if L ≤ koff2_lref\\l"
    "          = koff2 · koff2_lref / L           otherwise\\l"
    "\\l"
    "p_dloop(L) = 0                               if L < dloop_lmin\\l"
    "           = 1 / (1 + exp(-(L - dloop_l_half) / dloop_w))   otherwise\\l"
    "\\l"
    "extension bucket (i → i+1):  rate = kext / (L[i+1] - L[i])\\l"
    "    → kext is per-nt; rescaled by bucket gap so the discretization\\l"
    "      choice does not bias the kinetics.\\l"
    "\\l"
    "resection delay ~ Gamma(k = gamma_k, θ = gamma_theta)\\l"
    "    drawn per cell, applied as a flat segment (no solver call).\\l"
)
leg.node("laws", laws_text)

leg.edge("params", "laws", style="invis")  # vertical stack

leg_base = "gillespie_model_v2_legend"
for fmt in ("svg", "png", "pdf"):
    leg.render(leg_base, cleanup=True, format=fmt)
print(f"Wrote {leg_base}.svg/.png/.pdf")
