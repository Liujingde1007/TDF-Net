"""Generate the LaTeX table bodies from the experiment JSON and inject them into the .tex.

Transcribing 56 paired differences by hand is an error waiting to happen, so the table content is
generated instead.  The .tex carries placeholders

    %<TABLE_PAIRED>
    %<TABLE_MAIN>
    %<TABLE_OBS>

and this script replaces each with a freshly generated body, then checks that the result still
contains the headline numbers quoted in the prose.

Usage:  python code/make_latex_tables.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TEX = os.path.join(ROOT, "paper", "TDF-Net-ieeeCL.tex")

MARK = {"main": "%<TABLE_MAIN>", "paired": "%<TABLE_PAIRED>", "obs": "%<TABLE_OBS>"}

# display order of the estimators in Table I
MAIN_ORDER = [
    ("LS-linear", "LS + linear interp."),
    ("DFT-denoise", "DFT-denoise"),
    ("OMP", "OMP"),
    ("LMMSE", "LMMSE (sample stat.)"),
    ("ChannelNet", r"ChannelNet \cite{soltani2019}"),
    ("ResCNN", "ResCNN"),
    ("ResCNN+GRU", "ResCNN + GRU"),
    ("TDF-Net", r"\textbf{TDF-Net}"),
]

# rows of Table II: (source json, comparison key, label, reference label).  Labels stay short on
# purpose: the table carries nine columns, and long labels push it past the text block width.
PAIRED_ROWS = [
    ("paired_Ld_full", "Ld=1 - Ld=16", r"$L_d=1$", "$L_d=16$"),
    ("paired_Ld_full", "Ld=2 - Ld=16", r"$L_d=2$", "$L_d=16$"),
    ("paired_Ld_full", "Ld=4 - Ld=16", r"$L_d=4$", "$L_d=16$"),
    ("paired_Ld_full", "Ld=8 - Ld=16", r"$L_d=8$", "$L_d=16$"),
    ("paired_output", "out-Ld2 - feat-Ld2", r"output, $L_d=2$", r"feat., $L_d=2$"),
    ("paired_output", "out-Ld4 - feat-Ld2", r"output, $L_d=4$", r"feat., $L_d=2$"),
    ("paired_temporal", "M=8 - M=2", r"$M=8$", "$M=2$"),
    ("paired_temporal", "ChannelNet - M=2", r"ChannelNet \cite{soltani2019}", "$M=2$"),
]


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def g(d, k):
    return d[str(k)] if str(k) in d else d[k]


def fmt(v, star):
    s = "%+.2f" % v
    return "$%s%s$" % (s, "^{*}" if star else "")


def body_main(main):
    snrs = main["snr_db"]
    d = g(main["nmse_all"]["TDL-C"], 100.0)
    lines = []
    for key, label in MAIN_ORDER:
        cells = []
        for s in snrs:
            r = g(d, s)
            if key not in r:
                continue
            cells.append("$%.2f$" % r[key])
        if cells:
            lines.append("%-20s & %s \\\\" % (label, " & ".join(cells)))
    return "\n".join(lines)


def body_paired():
    snrs = load("paired_Ld_full")["snr_db"]
    lines = []
    for src, key, label, ref in PAIRED_ROWS:
        p = load(src)
        cells = []
        for s in snrs:
            rec = p["comparisons"][str(s)].get(key)
            if rec is None:
                break
            cells.append(fmt(rec["mean_db"], rec["significant"]))
        if len(cells) == len(snrs):
            lines.append("%-28s & %-12s & %s \\\\" % (label, ref, " & ".join(cells)))
    return "\n".join(lines)


def body_obs(rep, repf):
    rows = [
        ("full training distribution", r"de-rotated LS ($y/x$)", repf["derotated_nmse_db"]),
        ("full training distribution", r"raw sample ($y/\sigma$)", repf["raw_nmse_db"]),
        (r"fixed \SI{10}{\kilo\meter\per\hour}, \SI{15}{\decibel}",
         r"de-rotated LS ($y/x$)", rep["derotated_nmse_db"]),
        (r"fixed \SI{10}{\kilo\meter\per\hour}, \SI{15}{\decibel}",
         r"raw sample ($y/\sigma$)", rep["raw_nmse_db"]),
    ]
    return "\n".join("%s & %s & $%+.2f$ \\\\" % r for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify, do not write")
    args = ap.parse_args()

    text = open(TEX, encoding="utf-8").read()
    main_res = load("eval_main")
    rep = load("exp_input_representation")
    repf = load("exp_representation_full")

    bodies = {"main": body_main(main_res), "paired": body_paired(),
              "obs": body_obs(rep, repf)}

    unchanged = all(m in text for m in MARK.values())
    if unchanged and args.check:
        print("placeholders still present; run without --check to generate")
        return 1
    if not unchanged:
        print("no placeholders found; the .tex already carries generated tables")
        return 0

    for key, mark in MARK.items():
        text = text.replace(mark, bodies[key])
    with open(TEX, "w", encoding="utf-8") as f:
        f.write(text)

    n_rows = sum(len(b.splitlines()) for b in bodies.values())
    print("injected %d table rows into %s" % (n_rows, os.path.relpath(TEX, ROOT)))
    for k, b in bodies.items():
        print("  %-7s %d rows" % (k, len(b.splitlines())))

    # ---- verify that the headline numbers are still present in the prose
    checks = ["11.9", "19.2", "0.63", "0.59", "0.31", "2.30", "1.92", "4.26", "1.43",
              "-36.70", "-18.75", "-3.55", "-10.54", "11.43", "5.03", "753\\,298"]
    missing = [c for c in checks if c not in text]
    print("headline numbers present: %d/%d" % (len(checks) - len(missing), len(checks)))
    if missing:
        print("  MISSING: %s" % ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
