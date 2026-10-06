"""Deterministic figure placement.

1. Rewrites the figure references in the Results prose to the correct numbers.
2. Inserts each figure block at the end of the *paragraph* that cites it (after the blank line
   that terminates the paragraph), so no caption can split a paragraph.
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")

# final numbering: (number, file, role, caption)
FIGS = [
    (1, "figures/fig_nmse_tdlc_v30.png", "nmse_snr",
     "NMSE against SNR at 30 km/h (TDL-C), with identical training budgets, protocols and "
     "test sets for all learned estimators."),
    (2, "figures/fig_nmse_vs_velocity.png", "velocity",
     "NMSE at 25 dB against UE velocity. The ranking of the estimator families reverses "
     "between 30 and 100 km/h."),
    (3, "figures/fig_training.png", "training",
     "Training and validation NMSE of the three learned models. Validation tracks training "
     "throughout, so the comparisons below are not affected by over-fitting."),
    (4, "figures/fig_ablation.png", "ablation",
     "(a) Effect of the temporal window $M$ and (b) effect of removing the fixed delay-domain "
     "truncation branch."),
    (5, "figures/fig_delay_window.png", "window",
     "(a) The learned delay-domain window $w_d$ after training: it stays within 6 % of its "
     "initial value of unity for every retained tap. (b) The true discrete delay profile of "
     "the TDL-C channel used in the simulations."),
    (6, "figures/fig_ber_v100.png", "ber",
     "Post-equalisation BER at 100 km/h (TDL-C) with a genie-aided 1-tap MMSE equaliser. The "
     "differences between estimators are compressed relative to NMSE."),
    (7, "figures/fig_complexity.png", "complexity",
     "Model size and measured single-slot CPU inference time."),
]

# where each figure is first cited, identified by a unique phrase in that paragraph
CITE_ANCHOR = {
    "nmse_snr": "We organise the discussion by SNR regime",
    "velocity": "This is the most discriminating experiment in the study",
    "training": "all learned models converge smoothly",
    "ablation": "*Temporal window.* Table III and Fig. 4(a)",
    "window": "Second, the learned window stays within 6 %",
    "ber": "*Input representation.* Table IV isolates",
    "complexity": "TDF-Net has ",
}

# textual replacements that fix the references produced by fill_manuscript.py
REPLACEMENTS = [
    ("and Fig. 2 shows the curves", "and Fig. 1 shows the curves"),
    ("Fig. 3 plots NMSE against velocity", "Fig. 2 plots NMSE against velocity"),
    ("Fig. 1 shows the training and validation curves",
     "Fig. 3 shows the training and validation curves"),
    ("makes ResCNN+GRU worse than ResCNN above 15 dB (Fig. 3)",
     "makes ResCNN+GRU worse than ResCNN above 15 dB"),
    ("(Fig. 3). Temporal aggregation also costs memory", ". Temporal aggregation also costs memory"),
    ("TDF-Net has ", "The measured cost is summarised in Fig. 7. TDF-Net has "),
]


def main():
    src = os.path.join(PAPER, "manuscript_final.md")
    with open(src, encoding="utf-8") as f:
        text = f.read()

    for a, b in REPLACEMENTS:
        text = text.replace(a, b)

    paragraphs = text.split("\n\n")
    placed = set()
    out = []
    for para in paragraphs:
        out.append(para)
        for num, path, role, cap in FIGS:
            if role in placed:
                continue
            if not os.path.exists(os.path.join(ROOT, path)):
                continue
            anchor = CITE_ANCHOR[role]
            if anchor in para:
                out.append("![%s](%s)\n\n**Fig. %d.** %s"
                           % (cap.split(".")[0][:110], path, num, cap))
                placed.add(role)
                break
    text = "\n\n".join(out)

    dst = os.path.join(PAPER, "manuscript_illustrated.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    missing = [r for _, _, r, _ in FIGS if r not in placed]
    print("wrote %s with %d/%d figures" % (dst, len(placed), len(FIGS)))
    if missing:
        print("  not placed: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
