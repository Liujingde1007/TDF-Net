"""Figures and tables for the paired-difference results.

Produces
  figures/fig_paired_Ld.png/pdf       -- NMSE vs L_d with paired CIs against L_d = 16
  figures/fig_paired_ablation.png/pdf -- paired differences for the temporal window, the
                                         output-domain constraint and ChannelNet
  results/paired_tables.tex           -- LaTeX tables with mean +- SE and p-values
All numbers come from results/paired_*.json, produced by code/paired_test.py.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "axes.linewidth": 0.8, "savefig.bbox": "tight", "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote figures/%s.pdf|png" % name)


def fig_paired_ld(p):
    """NMSE against the truncation length, and paired differences vs L_d = 16."""
    snrs = p["snr_db"]
    names = p["names"]
    # absolute NMSE per configuration comes from paired_Ld_full (the comparison table)
    full = p
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    order = ["Ld=1", "Ld=2", "Ld=4", "Ld=8", "Ld=16"]
    for name in order:
        if name not in names:
            continue
        y = [full["comparisons"][str(s)][k]["mean_db"] for s in snrs for k in []]
    # panel (a): paired difference of each L_d against L_d = 16, with 95 % CI
    marks = {"Ld=1": ("o", "-"), "Ld=2": ("s", "--"), "Ld=4": ("^", "-."), "Ld=8": ("D", ":")}
    for name in ("Ld=1", "Ld=2", "Ld=4", "Ld=8"):
        key = "%s - Ld=16" % name
        if key not in full["comparisons"][str(snrs[0])]:
            continue
        m = [full["comparisons"][str(s)][key]["mean_db"] for s in snrs]
        lo = [full["comparisons"][str(s)][key]["ci95"][0] for s in snrs]
        hi = [full["comparisons"][str(s)][key]["ci95"][1] for s in snrs]
        mk, ls = marks[name]
        ax.plot(snrs, m, marker=mk, ls=ls, label="$L_d=%s$" % name.split("=")[1])
        ax.fill_between(snrs, lo, hi, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"SNR [dB]")
    ax.set_ylabel(r"NMSE difference vs $L_d=16$ [dB]")
    ax.set_title(r"(a) delay-domain branch strength")
    ax.legend(framealpha=0.9, loc="lower left")
    ax.text(0.02, 0.97, "negative = better", transform=ax.transAxes, fontsize=7,
            va="top", style="italic")

    # panel (b): paired differences for the output-domain constraint and temporal window
    po = load("paired_output")
    pt = load("paired_temporal")
    ax = axes[1]
    series = [
        ("output domain, $L_d=2$", po, "out-Ld2 - feat-Ld2", "o", "-"),
        ("output domain, $L_d=4$", po, "out-Ld4 - feat-Ld2", "s", "--"),
        ("temporal window $M=8$", pt, "M=8 - M=2", "^", "-."),
    ]
    for lab, src, key, mk, ls in series:
        m = [src["comparisons"][str(s)][key]["mean_db"] for s in snrs]
        lo = [src["comparisons"][str(s)][key]["ci95"][0] for s in snrs]
        hi = [src["comparisons"][str(s)][key]["ci95"][1] for s in snrs]
        ax.plot(snrs, m, marker=mk, ls=ls, label=lab)
        ax.fill_between(snrs, lo, hi, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"SNR [dB]")
    ax.set_ylabel(r"NMSE difference vs best feature model [dB]")
    ax.set_title(r"(b) constraint and window ablations")
    ax.legend(framealpha=0.9, loc="upper left")
    save(fig, "fig_paired_ablation")


def table_paired(p, ref, rows, caption, label):
    """LaTeX table of paired differences of several configurations against a reference."""
    snrs = p["snr_db"]
    out = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption,
           r"\label{%s}" % label, r"\begin{tabular}{l" + "c" * len(snrs) + "}", r"\hline",
           "Configuration & " + " & ".join("$%+g$" % s for s in snrs) + r" \\", r"\hline"]
    for name in rows:
        key = "%s - %s" % (name, ref)
        if key not in p["comparisons"][str(snrs[0])]:
            continue
        cells = []
        for s in snrs:
            r = p["comparisons"][str(s)][key]
            star = "$^{*}$" if r["significant"] else ""
            cells.append("%+.2f%s" % (r["mean_db"], star))
        out.append("%s & %s \\\\" % (name.replace("_", r"\_"), " & ".join(cells)))
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main():
    global fig_paired_ld
    p = load("paired_Ld_full")
    print("figures:")
    # absolute values for the L_d sweep are read from the comparison blocks of paired_Ld_full
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    snrs = p["snr_db"]
    absv = {n: [p["comparisons"][str(s)]["%s - Ld=16" % n]["mean_db"] for s in snrs]
            for n in ("Ld=1", "Ld=2", "Ld=4", "Ld=8")}
    # the reference itself is needed for absolute plotting; reconstruct from paired_temporal M=2
    ax = axes[0]
    for n in ("Ld=1", "Ld=2", "Ld=4", "Ld=8"):
        ax.plot(snrs, absv[n], marker="o", label="$L_d=%s$" % n.split("=")[1])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"SNR [dB]")
    ax.set_ylabel(r"NMSE difference vs $L_d=16$ [dB]")
    ax.set_title(r"(a) delay-domain branch strength")
    ax.legend(framealpha=0.9)
    ax.text(0.02, 0.97, "negative = better than $L_d=16$", transform=ax.transAxes,
            fontsize=7, va="top", style="italic")

    po = load("paired_output")
    pt = load("paired_temporal")
    ax = axes[1]
    for lab, src, key, mk in (("output domain, $L_d=2$", po, "out-Ld2 - feat-Ld2", "o"),
                              ("output domain, $L_d=4$", po, "out-Ld4 - feat-Ld2", "s"),
                              ("temporal window $M=8$", pt, "M=8 - M=2", "^")):
        m = [src["comparisons"][str(s)][key]["mean_db"] for s in snrs]
        lo = [src["comparisons"][str(s)][key]["ci95"][0] for s in snrs]
        hi = [src["comparisons"][str(s)][key]["ci95"][1] for s in snrs]
        ax.plot(snrs, m, marker=mk, label=lab)
        ax.fill_between(snrs, lo, hi, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"SNR [dB]")
    ax.set_ylabel(r"NMSE difference [dB]")
    ax.set_title(r"(b) constraint and window ablations")
    ax.legend(framealpha=0.9, loc="upper left")
    save(fig, "fig_paired_ablation")

    # ---- LaTeX tables
    tex = ["% auto-generated by code/make_paired_figures.py", ""]
    tex.append(table_paired(p, "Ld=16", ["Ld=1", "Ld=2", "Ld=4", "Ld=8"],
                            "Paired NMSE difference against $L_d=16$ [dB]; "
                            "negative is better. $^{*}$ marks 95 \\% significance.",
                            "tab:paired_ld"))
    tex.append("")
    tex.append(table_paired(po, "feat-Ld2", ["out-Ld2", "out-Ld4", "feat-Ld16", "feat-Ld4"],
                            "Paired NMSE difference against the best feature-level model "
                            "($L_d=2$) [dB].", "tab:paired_output"))
    tex.append("")
    tex.append(table_paired(pt, "M=2", ["M=8", "ChannelNet"],
                            "Paired NMSE difference against $M=2$ [dB].", "tab:paired_temporal"))
    with open(os.path.join(RESULTS, "paired_tables.tex"), "w") as f:
        f.write("\n".join(tex))
    print("  wrote results/paired_tables.tex")

    # ---- summary of the effect sizes that matter
    print("\nkey paired effects (dB, negative = first is better):")
    for src, key in ((p, "Ld=4 - Ld=16"), (p, "Ld=2 - Ld=16"),
                     (po, "out-Ld2 - feat-Ld2"), (po, "out-Ld4 - feat-Ld2"),
                     (pt, "M=8 - M=2"), (pt, "ChannelNet - M=2")):
        m = [src["comparisons"][str(s)][key]["mean_db"] for s in src["snr_db"]]
        sig = sum(1 for s in src["snr_db"] if src["comparisons"][str(s)][key]["significant"])
        print("  %-24s range %+.2f..%+.2f  significant at %d/%d SNR points"
              % (key, min(m), max(m), sig, len(m)))


if __name__ == "__main__":
    main()
