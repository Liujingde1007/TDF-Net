"""Build the compact IEEE Communications Letters version of the paper (Markdown/Word route).

Takes ``paper/manuscript_CL.md`` (the hand-written compressed text) and inserts the three data
tables from ``results/*.json``, then places the four figures.  All numbers come from the
evaluation JSON, exactly as for the full-length version.

SUBMISSION PATH.  ``paper/TDF-Net-ieeeCL.tex`` is now the hand-maintained submission source:
IEEEtran two-column, generated table bodies, appendix and author biography.  It is deliberately
NOT produced by this script, because a pandoc conversion cannot express the IEEEtran class, the
float placement, the appendix or the biography, and would overwrite the submission file with an
inferior one.  This script therefore writes only ``manuscript_CL_final.md`` and remains useful
for the plain-text/Word rendition of the same content.  Table bodies for the .tex come from
``code/make_latex_tables.py``; the source is validated by ``code/check_tex.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "results")

PRETTY = {
    "LS-linear": "LS + linear interp.",
    "DFT-denoise": "DFT-denoise",
    "OMP": "OMP",
    "LMMSE": "LMMSE (sample stat.)",
    "ChannelNet": "ChannelNet [5]",
    "ResCNN": "ResCNN",
    "ResCNN+GRU": "ResCNN + GRU",
    "TDF-Net": "**TDF-Net**",
}
ORDER = ["LS-linear", "DFT-denoise", "OMP", "LMMSE", "ChannelNet", "ResCNN", "ResCNN+GRU",
         "TDF-Net"]

FIGS = [
    ("figures/fig_nmse_tdlc_v30.png",
     "Fig. 1 shows the NMSE curves for all estimators at 30 km/h",
     "NMSE against SNR at 30 km/h (TDL-C) for all seven estimators, with identical training "
     "budgets, protocols and test sets."),
    ("figures/fig_nmse_vs_velocity.png",
     "Fig. 2 is the most discriminating experiment.",
     "NMSE at 25 dB against UE velocity. The ranking of the estimator families reverses "
     "between 30 and 100 km/h."),
    ("figures/fig_paired_ablation.png",
     "Fig. 3(a) and Table II report the truncation sweep",
     "Paired NMSE differences over 250 identical slots, with 95 % confidence bands. "
     "(a) Delay-domain truncation versus $L_d=16$; (b) the output-domain constraint and the "
     "temporal window versus the best feature-level model. Negative is better."),
]


def table_truncation(paired_ld, paired_out, paired_temp, snrs):
    """Table II: all ablations as paired differences against the appropriate reference."""
    rows = ["| Variant | Reference | " + " | ".join("%+g" % s for s in snrs) + " |",
            "|---" * (len(snrs) + 2) + "|"]
    for src, ref, keys in (
            (paired_ld, "Ld=16", [("Ld=1", "truncation $L_d=1$"),
                                  ("Ld=2", "truncation $L_d=2$"),
                                  ("Ld=4", "truncation $L_d=4$"),
                                  ("Ld=8", "truncation $L_d=8$")]),
            (paired_out, "feat-Ld2", [("out-Ld2", "output-constrained, $L_d=2$"),
                                      ("out-Ld4", "output-constrained, $L_d=4$")]),
            (paired_temp, "M=2", [("M=8", "temporal window $M=8$"),
                                  ("ChannelNet", "ChannelNet [5]")])):
        for name, lab in keys:
            key = "%s - %s" % (name, ref)
            if key not in src["comparisons"][str(snrs[0])]:
                continue
            cells = []
            for s in snrs:
                r = src["comparisons"][str(s)][key]
                cells.append("%+.2f%s" % (r["mean_db"], "*" if r["significant"] else ""))
            rows.append("| %s | %s | %s |" % (lab, ref, " | ".join(cells)))
    return ("**Paired NMSE difference against the stated reference [dB]; negative is better.** "
            "$^{*}$ marks $p<0.05$ over 250 identical slots.\n\n" + "\n".join(rows))


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def g(d, k):
    return d[str(k)] if str(k) in d else d[k]


def table_nmse(main, snrs, methods):
    d = g(main["nmse_all"]["TDL-C"], 100.0)
    rows = ["| Estimator | " + " | ".join("%+g" % s for s in snrs) + " |",
            "|---" * (len(snrs) + 1) + "|"]
    for m in methods:
        if m not in g(d, snrs[0]):
            continue
        rows.append("| %s | %s |" % (PRETTY.get(m, m),
                                     " | ".join("%.2f" % g(d, s)[m] for s in snrs)))
    return "\n".join(rows)


def table_obs(rep, repf):
    rows = ["| Protocol | Input representation | NMSE [dB] |", "|---|---|---|"]
    rows.append("| full training distribution | de-rotated LS ($y/x$) | %.2f |"
                % repf["derotated_nmse_db"])
    rows.append("| full training distribution | raw sample ($y/\\sigma$) | %+.2f |"
                % repf["raw_nmse_db"])
    rows.append("| fixed $v=%g$ km/h, SNR $=%g$ dB | de-rotated LS ($y/x$) | %.2f |"
                % (rep["velocity_kmh"], rep["snr_db"], rep["derotated_nmse_db"]))
    rows.append("| fixed $v=%g$ km/h, SNR $=%g$ dB | raw sample ($y/\\sigma$) | %+.2f |"
                % (rep["velocity_kmh"], rep["snr_db"], rep["raw_nmse_db"]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manuscript_CL.md")
    ap.add_argument("--out", default="manuscript_CL_final.md")
    args = ap.parse_args()

    text = open(os.path.join(PAPER, args.src), encoding="utf-8").read()
    main_res = load("eval_main")
    abl = load("eval_ablation")
    rep = load("eval_observation" if os.path.exists(os.path.join(RESULTS,
                                                "eval_observation.json"))
               else "exp_input_representation")
    repf = load("exp_representation_full")
    paired_ld = load("paired_Ld_full")
    paired_out = load("paired_output")
    paired_temp = load("paired_temporal")
    snrs = main_res["snr_db"]
    # the method list must come from the data, not from payload["methods"] (which only lists
    # the learned models actually evaluated)
    present = set(g(g(main_res["nmse_all"]["TDL-C"], 100.0), snrs[0]))
    methods = [m for m in ORDER if m in present]
    print("  methods found in eval_main: %s" % ", ".join(methods))

    # ---- tables: placed together after the conclusion, in citation order.  Journals set tables
    # near their first citation, but in a plain-Markdown master the safest place that cannot
    # split a paragraph is immediately before the reference list.
    tables = [
        ("**TABLE I**\n\n**NMSE [dB] on TDL-C at $v=100$ km/h.**\n\n"
         + table_nmse(main_res, snrs, methods)),
        ("**TABLE II**\n\n" + table_truncation(paired_ld, paired_out, paired_temp, snrs)),
    ]
    anchor = "## References"
    if anchor not in text:
        sys.exit("no '## References' section to anchor the tables")
    text = text.replace(anchor, "\n\n".join(tables) + "\n\n---\n\n" + anchor, 1)

    # ---- figures: inserted at the end of the paragraph that first cites them
    placed = 0
    for img, cite, cap in FIGS:
        if not os.path.exists(os.path.join(ROOT, img)):
            print("  [skip] missing %s" % img)
            continue
        pat = re.compile(r"\s+".join(map(re.escape, cite.split())))
        m = pat.search(text)
        if not m:
            print("  [warn] figure citation not found: %r" % cite[:60])
            continue
        placed += 1
        end = text.find("\n\n", m.end())
        end = len(text) if end < 0 else end
        text = (text[:end]
                + "\n\n![%s](%s)\n\n**Fig. %d.** %s"
                % (cap.split(".")[0][:110], img, placed, cap)
                + text[end:])

    dst = os.path.join(PAPER, args.out)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    words = len(text.split())
    print("wrote %s  (%d words, %d figures, %d tables)" % (dst, words, placed, len(tables)))
    # IEEEtran two-column, 10 pt: roughly 1100 words per full page of body text.  Figures and
    # tables are counted as fractions of a page, generously.
    body_pp = words / 1100.0
    float_pp = placed * 0.22 + len(tables) * 0.15
    print("  rough length: %.1f pages body + %.1f pages floats = %.1f of 5"
          % (body_pp, float_pp, body_pp + float_pp))


if __name__ == "__main__":
    main()
