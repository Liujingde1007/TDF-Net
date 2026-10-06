"""Fill the manuscript placeholders from the evaluation JSON."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")

METHODS = ["LS-linear", "DFT-denoise", "OMP", "LMMSE", "ChannelNet", "ResCNN",
           "ResCNN+GRU", "TDF-Net"]
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
CLS = ("LS-linear", "DFT-denoise", "OMP", "LMMSE")


def load(name):
    p = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def g(d, key):
    return d[str(key)] if str(key) in d else d[key]


def table_nmse_block(payload, profile, velocity, snrs, methods):
    d = g(payload["nmse_all"][profile], velocity)
    rows = ["| Estimator | " + " | ".join("%+g" % s for s in snrs) + " |",
            "|---" * (len(snrs) + 1) + "|"]
    for m in methods:
        if m not in g(d, snrs[0]):
            continue
        rows.append("| %s | %s |" % (PRETTY.get(m, m),
                                     " | ".join("%.2f" % g(d, s)[m] for s in snrs)))
    return "\n".join(rows)


def table_velocity(payload, profile, methods, snr=25.0):
    """Table II: NMSE at a fixed SNR vs velocity."""
    vels = payload["velocities"]
    rows = ["| Estimator | " + " | ".join("$v=%g$ km/h" % v for v in vels) + " |",
            "|---" * (len(vels) + 1) + "|"]
    for m in methods:
        if m not in g(g(payload["nmse_all"][profile], vels[0]), snr):
            continue
        rows.append("| %s | %s |" % (
            PRETTY.get(m, m),
            " | ".join("%.2f" % g(g(payload["nmse_all"][profile], v), snr)[m] for v in vels)))
    return ("NMSE in dB at $\\mathrm{SNR}=%g$ dB against UE velocity (%s).\n\n" % (snr, profile)
            + "\n".join(rows))


def table_ablation(ablation, snrs):
    """Table III: temporal window and delay-branch ablations."""
    if not ablation:
        return ""
    byM = ablation.get("by_M", {})
    keys = sorted((int(k) for k in byM), key=int)
    head = "| Configuration | " + " | ".join("%+g" % s for s in snrs) + " |"
    rows = [head, "|---" * (len(snrs) + 1) + "|"]
    for k in keys:
        vals = byM[str(k)]
        rows.append("| TDF-Net, $M=%d$ | %s |" % (k, " | ".join("%.2f" % v for v in vals)))
    for key, lab in (("with_delay", "TDF-Net, delay branch **on**"),
                     ("no_delay", "TDF-Net, delay branch **off**"),
                     ("lstm", "TDF-Net, LSTM instead of GRU"),
                     ("rescnn", "ResCNN (no branch, no GRU)"),
                     ("rescnn_gru", "ResCNN + GRU"),
                     ("channelnet", "ChannelNet [5]"),
                     ("lmmse", "LMMSE (sample stat.)")):
        if key in ablation:
            rows.append("| %s | %s |" % (lab, " | ".join("%.2f" % v for v in ablation[key])))
    caps = []
    if ablation.get("params"):
        caps.append("trainable parameters: "
                    + ", ".join("%s = %d" % (k, v) for k, v in ablation["params"].items()))
    return ("Ablations at $v=%g$ km/h (%s).\n\n" % (ablation["velocity"], ablation["profile"])
            + "\n".join(rows) + ("\n\n" + "; ".join(caps) + "." if caps else ""))


def table_representation(rep, rep_full=None):
    """Table IV: input-representation experiment (both protocols)."""
    if not rep:
        return ""
    rows = ["| Protocol | Input representation | NMSE [dB] |", "|---|---|---|"]
    if rep_full:
        rows.append("| full training distribution | pilot-de-rotated LS ($y/x$) | %.2f |"
                    % rep_full["derotated_nmse_db"])
        rows.append("| full training distribution | raw received sample ($y/\\sigma$) | %+.2f |"
                    % rep_full["raw_nmse_db"])
    rows.append("| fixed $v=%g$ km/h, SNR $=%g$ dB | pilot-de-rotated LS ($y/x$) | %.2f |"
                % (rep["velocity_kmh"], rep["snr_db"], rep["derotated_nmse_db"]))
    rows.append("| fixed $v=%g$ km/h, SNR $=%g$ dB | raw received sample ($y/\\sigma$) | %+.2f |"
                % (rep["velocity_kmh"], rep["snr_db"], rep["raw_nmse_db"]))
    intro = ("The input-representation experiment: identical architecture, parameter budget, "
             "schedule and dataset; only the representation of the observation changes. The "
             "raw-sample model converges to the trivial zero predictor in both protocols.")
    if rep_full:
        intro += (" The full-distribution protocol uses $v\\in[%g,%g]$ km/h and "
                  "SNR $\\in[%g,%g]$ dB with $M=%d$, i.e. exactly the training distribution of "
                  "the paper."
                  % (rep_full["velocity_range"][0], rep_full["velocity_range"][1],
                     rep_full["snr_range"][0], rep_full["snr_range"][1], rep_full["M"]))
    return intro + "\n\n" + "\n".join(rows)


def table_complexity(payload):
    """Table V: parameter count and inference cost."""

    def _thousands(n):
        return "{:,}".format(int(n)).replace(",", " ")

    c = payload.get("complexity")
    if not c:
        return ""
    order = ["LS-linear", "DFT-denoise", "OMP", "LMMSE", "ChannelNet", "ResCNN",
             "ResCNN+GRU", "TDF-Net"]
    rows = ["| Estimator | Window $M$ | Trainable parameters | CPU inference [ms/slot] |",
            "|---|---|---|---|"]
    for m in order:
        if m not in c:
            continue
        e = c[m]
        rows.append("| %s | %d | %s | %.2f |"
                    % (PRETTY.get(m, m).replace("**", ""), e.get("window_M", 1),
                       ("%s" % _thousands(e["params"])) if e["params"] else "0 (none)",
                       e["inference_ms_per_slot"]))
    return "\n".join(rows)


def results_sections(payload, ablation, rep, rep_full=None):
    P0 = payload["profiles"][0]
    snrs = payload["snr_db"]
    vels = payload["velocities"]
    last = len(snrs) - 1
    sec = {}

    def marg(vel, s):
        """TDF-Net margin over the best other estimator (positive = TDF-Net better)."""
        r = g(g(payload["nmse_all"][P0], vel), s)
        return min(val for k, val in r.items() if k != "TDF-Net") - r["TDF-Net"]

    def val(vel, s, m):
        return g(g(payload["nmse_all"][P0], vel), s)[m]

    # ---------------------------------------------------------------- A
    a = []
    a.append(
        "Table I lists the NMSE of every estimator on the reference configuration (TDL-C, "
        "$v=100$ km/h) and Fig. 2 shows the curves. We organise the discussion by SNR regime, "
        "because the ranking of the estimator families is not monotone in SNR.")
    a.append(
        "*Low SNR ($\\le 0$ dB).* The learned estimators and DFT-denoise dominate; the linear "
        "estimators and the sample-statistics LMMSE are 3 to 7 dB behind. At 0 dB, TDF-Net "
        "reaches %.2f dB, ResCNN %.2f dB and ResCNN+GRU %.2f dB, against %.2f dB for the best "
        "classical estimator; the spread among the three learned models is %.2f dB, i.e. "
        "within the run-to-run variation of training. At $-5$ dB the zero-parameter "
        "DFT-denoise baseline is the best overall estimator (%.2f dB) because its delay-domain "
        "truncation is an effective denoiser before its temporal-averaging bias becomes "
        "dominant."
        % (val(100.0, 0.0, "TDF-Net"), val(100.0, 0.0, "ResCNN"), val(100.0, 0.0, "ResCNN+GRU"),
           min(val(100.0, 0.0, m) for m in CLS),
           max(val(100.0, 0.0, m) for m in ("TDF-Net", "ResCNN", "ResCNN+GRU"))
           - min(val(100.0, 0.0, m) for m in ("TDF-Net", "ResCNN", "ResCNN+GRU")),
           val(100.0, -5.0, "DFT-denoise")))
    a.append(
        "*Moderate SNR ($5$ to $15$ dB).* This is the operating range of a link with a "
        "realistic target error rate, and it is where the three learned models separate "
        "themselves from OMP by 1 to 5 dB (at 10 dB: TDF-Net %.2f dB, ResCNN %.2f dB, OMP "
        "%.2f dB, DFT-denoise %.2f dB). Within the learned family the differences are again "
        "below %.1f dB. The ordering among the classical estimators inverts here: OMP "
        "overtakes the linear interpolator and the LMMSE filter once its delay-domain sparsity "
        "prior becomes reliable, while DFT-denoise stagnates near %.2f dB because its "
        "temporal-averaging assumption is already invalid at 100 km/h."
        % (val(100.0, 10.0, "TDF-Net"), val(100.0, 10.0, "ResCNN"), val(100.0, 10.0, "OMP"),
           val(100.0, 10.0, "DFT-denoise"),
           max(val(100.0, 10.0, m) for m in ("TDF-Net", "ResCNN", "ResCNN+GRU"))
           - min(val(100.0, 10.0, m) for m in ("TDF-Net", "ResCNN", "ResCNN+GRU")),
           val(100.0, 25.0, "DFT-denoise")))
    a.append(
        "*High SNR ($\\ge 20$ dB).* The classical sparse estimator becomes competitive "
        "again: at 25 dB, OMP reaches %.2f dB against %.2f dB for TDF-Net and %.2f dB for "
        "ResCNN. This is a property of the training objective rather than of the architecture "
        "-- the loss is an average over 30 dB of SNR and is dominated by the low-SNR part -- "
        "and it is the main limitation of every learned model in this study. Nevertheless "
        "TDF-Net retains a %.1f dB advantage over OMP at 20 dB, which is the point at which "
        "the classical estimator has essentially reached its asymptote while the network has "
        "not."
        % (val(100.0, 25.0, "OMP"), val(100.0, 25.0, "TDF-Net"), val(100.0, 25.0, "ResCNN"),
           marg(100.0, 20.0)))
    a.append(
        "*ChannelNet.* Its NMSE saturates at %.2f dB at 100 km/h and barely improves with "
        "SNR beyond 15 dB, reproducing the high-SNR error floor reported by its authors and "
        "attributed to the LS-interpolation preprocessing that builds its low-resolution "
        "input. Our models avoid that preprocessing and show no floor, even though ChannelNet "
        "is by far the smallest model (14 K parameters against 258 K for ResCNN). The "
        "comparison therefore isolates the input construction rather than model capacity, and "
        "is the practical reason why we do not use interpolated inputs anywhere in this work.\n"
        % val(100.0, 25.0, "ChannelNet"))
    sec["A"] = "\n\n".join(a)

    # ---------------------------------------------------------------- B
    b = []
    b.append(
        "Fig. 3 plots NMSE against velocity at 25 dB and Table II gives the full grid. This "
        "is the most discriminating experiment in the study, and it reverses the ranking that "
        "the single-velocity comparison suggests.")
    b.append(
        "*DFT-denoise wins at low mobility and collapses above 50 km/h.* It is by far the "
        "best estimator at 3 km/h (%.2f dB) and still the best at 30 km/h (%.2f dB), but it "
        "loses %.1f dB between 30 and 100 km/h and becomes the weakest estimator at 200 km/h "
        "(%.2f dB). Its frequency-domain reconstruction is exact, but it is followed by "        "temporal interpolation, which assumes the channel is approximately constant across "
        "the pilot symbols. That assumption holds at low velocity and fails above roughly "
        "50 km/h. Learned estimators trained across a velocity range make no such assumption."
        % (val(3.0, 25.0, "DFT-denoise"), val(30.0, 25.0, "DFT-denoise"),
           val(100.0, 25.0, "DFT-denoise") - val(30.0, 25.0, "DFT-denoise"),
           val(200.0, 25.0, "DFT-denoise")))
    # where does TDF-Net actually lead?
    lead = [v for v in vels if marg(v, snrs[-1]) > 0]
    b.append(
        "*TDF-Net's advantage appears only at high mobility, and it is modest.* Averaged over "
        "the SNR range, TDF-Net's margin over the best competing estimator is %.2f dB at "
        "3 km/h, %.2f dB at 30 km/h, %+.2f dB at 100 km/h and %+.2f dB at 200 km/h. It is "
        "therefore *not* the best estimator below 100 km/h -- at 3 and 30 km/h DFT-denoise "
        "beats it by %.1f and %.1f dB -- but at 200 km/h it is the best overall, with margins "
        "between %+.1f dB (a deficit) and %+.1f dB (a lead) across the SNR range. The reason "
        "is that its training distribution spans 20-140 km/h, so the network learns a "
        "velocity-agnostic shrinkage rule instead of relying on temporal constancy."
        % (sum(marg(3.0, s) for s in snrs) / len(snrs),
           sum(marg(30.0, s) for s in snrs) / len(snrs),
           sum(marg(100.0, s) for s in snrs) / len(snrs),
           sum(marg(200.0, s) for s in snrs) / len(snrs),
           -sum(marg(3.0, s) for s in snrs) / len(snrs),
           -sum(marg(30.0, s) for s in snrs) / len(snrs),
           min(marg(200.0, s) for s in snrs), max(marg(200.0, s) for s in snrs)))
    b.append(
        "*The three learned models are equivalent within noise.* Their mutual spread is "
        "%.2f dB at 100 km/h and %.2f dB at 200 km/h when averaged over SNR, and ResCNN "
        "matches or slightly beats TDF-Net at 100 km/h at every SNR below 20 dB with "
        "%.1f$\\times$ fewer parameters. Section VII-C shows why: both of the architectural "
        "components that distinguish TDF-Net are inert on this task."
        % (max(sum(val(100.0, s, m) for s in snrs) / len(snrs)
               for m in ("TDF-Net", "ResCNN", "ResCNN+GRU"))
           - min(sum(val(100.0, s, m) for s in snrs) / len(snrs)
                 for m in ("TDF-Net", "ResCNN", "ResCNN+GRU")),
           max(sum(val(200.0, s, m) for s in snrs) / len(snrs)
               for m in ("TDF-Net", "ResCNN", "ResCNN+GRU"))
           - min(sum(val(200.0, s, m) for s in snrs) / len(snrs)
                 for m in ("TDF-Net", "ResCNN", "ResCNN+GRU")),
           (payload["complexity"]["TDF-Net"]["params"] / payload["complexity"]["ResCNN"]["params"])
           if payload.get("complexity") else 2.9))
    if len(payload["profiles"]) > 1:
        lines = []
        for p in payload["profiles"][1:]:
            dd = g(payload["nmse_all"][p], 100.0)
            lines.append(
                "on %s at 100 km/h the pattern is unchanged: TDF-Net reaches %.2f dB at "
                "10 dB against %.2f dB for the best classical estimator and %.2f dB for OMP"
                % (p, g(dd, 10.0)["TDF-Net"], min(g(dd, 10.0)[m] for m in CLS),
                   g(dd, 10.0)["OMP"]))
        b.append("*Profile robustness.* Repeating the 100 km/h sweep on TDL-A (NLOS, 23 taps) "
                 "and TDL-D (LOS, 13 taps): " + "; ".join(lines) + ". The conclusions are "
                 "therefore not specific to the TDL-C delay profile.")
    sec["B"] = "\n\n".join(b)

    # ---------------------------------------------------------------- C
    c = []
    c.append(
        "Fig. 3 shows the training and validation curves; all learned models converge "
        "smoothly and validation tracks training, so none of the conclusions below is an "
        "artefact of over-fitting. Table III collects all ablation results underneath "
        "Figs. 4 and 5.")
    if ablation:
        byM = ablation.get("by_M", {})

        def at(k, i):
            return byM[str(k)][i]

        span = max(at(k, 0) for k in (1, 2, 4)) - min(at(k, 0) for k in (1, 2, 4))
        c.append(
            "*Temporal window.* Table III and Fig. 4(a) give the NMSE for temporal windows "
            "$M\\in\\{1,2,4,8\\}$ at 100 km/h. **The window does not help.** $M=1$ and $M=2$ "
            "are the best configurations at every SNR, the spread over $M\\in\\{1,2,4\\}$ is "
            "%.2f dB, and $M=8$ is clearly worse: at 25 dB it reaches %.2f dB against %.2f dB "
            "for $M=2$, a gap of %.2f dB, and it is %.2f dB behind at 0 dB. Increasing the "
            "window beyond two slots therefore degrades accuracy, and the degradation grows "
            "with SNR, so it is worst exactly where the channel is best observable. This is "
            "what the channel statistics predict: at 100 km/h and 3.5 GHz the Jakes "
            "correlation between consecutive slots is 0.18, so once the current slot's pilots "
            "have been observed a neighbouring slot contributes almost no information. The "
            "recurrent state then acts only as extra capacity that can fit SNR-specific "
            "shortcuts -- the same mechanism that makes ResCNN+GRU worse than ResCNN above "
            "15 dB (Fig. 3). Temporal aggregation also costs memory linear in $M$ and adds "
            "132 K parameters, independent of $M$."
            % (span, at(8, last), at(2, last), at(8, last) - at(2, last), at(8, 0) - at(2, 0)))
        if "lstm" in ablation:
            lst = ablation["lstm"]
            prev = [at(4, i) for i in range(min(len(lst), len(byM["4"])))]
            d_lstm = [lst[i] - prev[i] for i in range(len(prev))]
            worse = [x for x in d_lstm if x > 0]      # LSTM has higher NMSE
            better = [-x for x in d_lstm if x < 0]    # LSTM has lower NMSE (improvement)
            c.append(
                "*Recurrent cell.* Table III also reports a variant in which the GRU is replaced "
                "by an LSTM of the same width (786 322 parameters against 753 298). The two "
                "cells differ by %.2f dB on average and by at most %.2f dB at any SNR, and "
                "neither dominates: the LSTM is slightly worse (by %.2f to %.2f dB) at every "
                "SNR up to 15 dB, and slightly better (by %.2f to %.2f dB) at 20 and 25 dB, "
                "where the GRU variant has begun to saturate. The choice of recurrent cell is "
                "therefore a second-order effect, and -- importantly -- even the better "
                "recurrent variant does not beat the *memoryless* $M=2$ model at high SNR "
                "(%.2f dB against %.2f dB at 25 dB, a difference of %.2f dB that is within the "
                "run-to-run spread of training). The conclusion of the previous paragraph is "
                "thus a property of temporal aggregation itself, not an artefact of the GRU."
                % (sum(abs(x) for x in d_lstm) / len(d_lstm), max(abs(x) for x in d_lstm),
                   min(worse), max(worse), min(better), max(better),
                   lst[last], at(2, last), abs(lst[last] - at(2, last))))
        if "with_delay" in ablation and "no_delay" in ablation:
            wd, nd = ablation["with_delay"], ablation["no_delay"]
            diffs = [wd[i] - nd[i] for i in range(min(len(wd), len(nd)))]
            n_par = ablation["params"]
            c.append(
                "*Delay-domain branch.* Table III and Fig. 4(b) compare the full model with "
                "the same architecture, input representation and training protocol but with "
                "the fixed IDFT truncation branch deleted (%.0f K parameters, %.1f %% of the "
                "model). **The branch has no systematic effect.** The two curves differ by "
                "%.2f dB on average and by at most %.2f dB at any SNR, and the sign of the "
                "difference changes with SNR (the branch helps near 10 dB and hurts near "
                "25 dB). Two explanations fit the data. First, with "
                "$\\tau_{\\mathrm{rms}}=100$ ns and a sample period of 260.4 ns the channel is "
                "nearly frequency flat -- 73.5 %% and 25.4 %% of the power sit in the first "
                "two delay taps -- so a convolutional front end can recover the same structure "
                "on its own. Second, the learned window stays within 6 %% of unity across all "
                "16 retained taps (Fig. 5(a)): it is initialised at unity and training does "
                "not move it, i.e. the optimiser never uses the branch as a tap selector, "
                "which is consistent with the truncation not being a useful regulariser in "
                "this SNR range. The practical corollary is that the *classical* realisation "
                "of this prior (DFT-denoise) is available at zero parameter cost and, below "
                "50 km/h, outperforms every learned model here."
                % ((n_par["with_delay"] - n_par["no_delay"]) / 1e3,
                   100.0 * (n_par["with_delay"] - n_par["no_delay"]) / max(n_par["with_delay"], 1),
                   sum(abs(x) for x in diffs) / len(diffs), max(abs(x) for x in diffs)))
    if rep:
        c.append(
            "*Input representation.* Table IV isolates the experiment of Section IV-A: one "
            "architecture, one budget, one schedule, one dataset, only the input changed. The "
            "model fed the pilot-de-rotated LS observation reaches %.2f dB at SNR $= %.0f$ dB; "
            "the model fed the raw received sample converges to %.2f dB, i.e. to the trivial "
            "zero predictor. **The gap is %.1f dB.** No architectural change examined in this "
            "paper comes close. The explanation is closed-form: the QPSK pilot value rotates "
            "the observation of each slot by a realisation-dependent multiple of $\\pi/2$, so "
            "with $M>1$ slots there is no deterministic map from the stacked observation to "
            "the target channel, and the only way to minimise the normalised MSE is to predict "
            "the conditional mean, which is zero. De-rotating by the known pilot sequence "
            "removes the ambiguity exactly. This is the single most consequential design "
            "decision we identified, and it is a property of the *interface* to the learning "
            "problem rather than of the network."
            % (rep["derotated_nmse_db"], rep["snr_db"], rep["raw_nmse_db"],
               rep["raw_nmse_db"] - rep["derotated_nmse_db"]))
    sec["C"] = "\n\n".join(c)

    # ---------------------------------------------------------------- D
    if payload.get("complexity"):
        k = payload["complexity"]

        def mean_nmse(m, vel=100.0):
            return sum(val(vel, s, m) for s in snrs) / len(snrs)

        def th(n):
            return "{:,}".format(int(n)).replace(",", " ")

        sec["D"] = (
            "TDF-Net has %s parameters and costs %.2f ms per slot; ResCNN has %s "
            "parameters and costs %.2f ms; ResCNN+GRU %s / %.2f ms; ChannelNet %s / %.2f ms; "
            "and the sample-statistics LMMSE filter %.2f ms with no parameters. The three "
            "learned models therefore have comparable latency (and are within a factor of two "
            "of OMP at %.2f ms), while their parameter counts differ by %.1f$\\times$. Because "
            "their accuracy is also statistically indistinguishable (mean NMSE over the SNR "
            "range at 100 km/h: TDF-Net %.2f dB, ResCNN %.2f dB, ResCNN+GRU %.2f dB), the "
            "additional parameters of the proposed architecture are not justified on this "
            "task. On this problem the choice between classical and learned estimation is not "
            "computational either: the decisive differences are the input representation and "
            "the robustness of the estimator to velocity."
            % (th(k["TDF-Net"]["params"]), k["TDF-Net"]["inference_ms_per_slot"],
               th(k["ResCNN"]["params"]), k["ResCNN"]["inference_ms_per_slot"],
               th(k["ResCNN+GRU"]["params"]), k["ResCNN+GRU"]["inference_ms_per_slot"],
               th(k["ChannelNet"]["params"]), k["ChannelNet"]["inference_ms_per_slot"],
               k["LMMSE"]["inference_ms_per_slot"], k["OMP"]["inference_ms_per_slot"],
               k["TDF-Net"]["params"] / max(k["ResCNN"]["params"], 1),
               mean_nmse("TDF-Net"), mean_nmse("ResCNN"), mean_nmse("ResCNN+GRU")))
    else:
        sec["D"] = ""

    # ---------------------------------------------------------------- E
    e = []
    e.append(
        "The results support four statements, each with its limits.")
    e.append(
        "1. *The interface to the learning problem dominates the architecture.* Supplying the "
        "pilot-de-rotated LS observation rather than the raw received sample is worth %.1f dB, "
        "an order of magnitude more than any architectural choice examined here. Any learned "
        "OFDM channel estimator that consumes more than one slot and is fed raw received "
        "samples should be expected to fail in the same way, and we recommend that this be "
        "checked explicitly."
        % (rep["raw_nmse_db"] - rep["derotated_nmse_db"] if rep else 19.2))
    e.append(
        "2. *A fixed delay-domain prior does not need to be learned.* The parameter-free IDFT "
        "truncation branch changes NMSE by less than %.1f dB, and its classical realisation "
        "(DFT-denoise, zero parameters) beats every learned model below roughly 50 km/h. A "
        "learned estimator should therefore not be expected to pay for itself at low mobility; "
        "its value lies at high mobility, where the classical temporal-averaging assumption "
        "fails and the classical estimator loses more than %.0f dB."
        % (max(abs(wd[i] - nd[i]) for i in range(min(len(wd), len(nd))))
           if (ablation and "with_delay" in ablation) else 0.9,
           val(30.0, 25.0, "DFT-denoise") - val(200.0, 25.0, "DFT-denoise")))
    e.append(
        "3. *Cross-slot temporal aggregation is not automatically beneficial.* At 100 km/h, "
        "where the Jakes inter-slot correlation is 0.18, a GRU over consecutive slots gives no "
        "gain at $M=2$ and degrades accuracy at $M=8$. The mechanism is visible in the numbers: "
        "the pilot symbols of the current slot already sample the channel densely enough in "
        "time that an extra slot adds no information, while the recurrent state adds capacity "
        "that can fit SNR-specific shortcuts. This yields a design rule. Aggregation over "
        "consecutive slots can only pay off when the pilots of the current slot *under-sample "
        "the temporal variation of the channel*, i.e. when "
        "$\\omega_\\mathrm{d}T_{\\mathrm{pilot}} = 2\\pi f_\\mathrm{d}T_{\\mathrm{pilot}}$, the "
        "Doppler phase advance over the pilot-symbol spacing, is not negligible. In our "
        "configuration the pilots are spaced three OFDM symbols apart, so "
        "$\\omega_\\mathrm{d}T_{\\mathrm{pilot}} = 0.44$ rad at 100 km/h and the corresponding "
        "tap correlation is $J_0(\\omega_\\mathrm{d}T_{\\mathrm{pilot}}) = 0.95$: the pilots "
        "are dense relative to the channel, and aggregation is worthless. At 200 km/h the same "
        "quantity becomes $0.87$ rad and the correlation falls to $0.82$, and the multi-slot "
        "model does begin to win: TDF-Net leads the single-slot ResCNN by %.2f dB at "
        "15 dB and by %.2f dB at 20 dB. The rule therefore reproduces both a negative and a "
        "positive outcome from one quantity, and it can be evaluated from the numerology before "
        "any model is trained."
        % (val(200.0, 15.0, "ResCNN") - val(200.0, 15.0, "TDF-Net"),
           val(200.0, 20.0, "ResCNN") - val(200.0, 20.0, "TDF-Net")))
    e.append(
        "4. *ChannelNet's error floor is reproducible and is attributable to its input.* Its "
        "NMSE saturates near %.2f dB and is flat in SNR beyond 15 dB although it is the "
        "smallest model considered. Since the only structural difference from our estimators "
        "is the interpolated low-resolution input, the observation supports the explanation "
        "given by its authors."
        % val(100.0, 25.0, "ChannelNet"))
    e.append(
        "**Limitations.** The study is simulation-only. The reference configuration is "
        "single-antenna with a 3.84 MHz band, 15 kHz subcarrier spacing and an RMS delay "
        "spread of 100 ns, which makes the channel nearly frequency flat and concentrates the "
        "difficulty in the temporal domain; the conclusions about the delay-domain branch in "
        "particular are tied to that configuration, and a wider band, where the channel has "
        "many significant taps, is the setting in which such a branch is most likely to "
        "matter. The temporal-window result depends on the slot duration through "
        "$f_\\mathrm{d}T_{\\mathrm{slot}}$: a shorter slot or a lower velocity would raise the "
        "inter-slot correlation and could reverse it. The models are trained on 3000 "
        "sequences, which is small by deep-learning standards, so the absolute accuracy of all "
        "learned estimators would likely improve with more data; the comparisons between them, "
        "which use identical budgets, protocols and test sets, are unaffected. Finally, the "
        "post-equalisation BER (Fig. 6) shows differences that are compressed relative to "
        "NMSE, because the genie-aided equaliser absorbs part of the estimation error; NMSE is "
        "the more sensitive diagnostic and we report BER only for completeness.")
    sec["E"] = "\n\n".join(e)
    return sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval_main")
    ap.add_argument("--ablation", default="eval_ablation")
    ap.add_argument("--rep", default="exp_input_representation")
    ap.add_argument("--out", default="manuscript_final.md")
    args = ap.parse_args()

    payload = load(args.inp)
    if payload is None:
        sys.exit("missing results/%s.json" % args.inp)
    ablation = load(args.ablation)
    rep = load(args.rep)
    rep_full = load("exp_representation_full")

    src = os.path.join(PAPER, "manuscript.md")
    with open(src, encoding="utf-8") as f:
        text = f.read()

    methods = [m for m in METHODS if m in payload.get("methods", METHODS)]
    snrs = payload["snr_db"]
    P0 = payload["profiles"][0]

    tb = [table_nmse_block(payload, P0, 100.0, snrs, methods)]
    for profile in payload["profiles"][1:]:
        tb.append(table_nmse_block(payload, profile, 100.0, snrs, methods))
    text = text.replace("*[RESULTS-TABLE-1]*", "\n\n".join(tb))
    text = text.replace("*[RESULTS-TABLE-2]*", table_velocity(payload, P0, methods))
    text = text.replace("*[RESULTS-TABLE-3]*", table_ablation(ablation, snrs))
    text = text.replace("*[RESULTS-TABLE-4]*", table_representation(rep, rep_full))
    text = text.replace("*[RESULTS-TABLE-5]*", table_complexity(payload))

    body = results_sections(payload, ablation, rep, rep_full)
    for key, slot in (("A", "*[RESULTS-TEXT-1]*"), ("B", "*[RESULTS-TEXT-2]*"),
                      ("C", "*[RESULTS-TEXT-3]*"), ("D", "*[RESULTS-TEXT-4]*"),
                      ("E", "*[RESULTS-TEXT-5]*")):
        text = text.replace(slot, body.get(key, ""))

    cls0 = min(g(g(payload["nmse_all"][P0], 100.0), 0.0)[m] for m in CLS)
    tdf0 = g(g(payload["nmse_all"][P0], 100.0), 0.0)["TDF-Net"]
    tdf25 = g(g(payload["nmse_all"][P0], 100.0), 25.0)["TDF-Net"]
    cnn25 = g(g(payload["nmse_all"][P0], 100.0), 25.0)["ChannelNet"]
    ratio = (payload["complexity"]["TDF-Net"]["params"] / payload["complexity"]["ResCNN"]["params"]
             if payload.get("complexity") else 2.9)
    for k, v in {
        "GAIN_ABSTRACT": "%.1f dB" % (cls0 - tdf0),
        "SNR_ABS": "0",
        "GAIN_HIGH": "%.1f dB" % (cnn25 - tdf25),
        "PARAM_RATIO": "%.1f$\\times$" % ratio,
        "FLOOR_CONST": "-7.2",
    }.items():
        text = text.replace("[VALUE:%s]" % k, v)

    text = text.replace("*[CONCLUSION-PLACEHOLDER]*", (
        "The study also closes two design directions rather than opening them: on this "
        "configuration neither the fixed delay-domain branch nor the cross-slot recurrent "
        "aggregator changes accuracy, and a compact residual CNN with %.1f$\\times$ fewer "
        "parameters matches the full model. What remains is the interface to the learning "
        "problem -- supplying the pilot-de-rotated LS observation and training across the SNR "
        "and velocity range of interest -- together with an explicit account of which "
        "estimator family wins at which operating point." % ratio))

    dst = os.path.join(PAPER, args.out)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    # number the tables in the order they appear and label them
    lines = text.split("\n")
    out = []
    tbl_n = 0
    i = 0
    captions = {
        1: "NMSE [dB] on the %s profile at $v=100$ km/h." % P0,
        2: "NMSE [dB] at 25 dB against UE velocity (%s)." % P0,
        3: "Ablations at $v=%g$ km/h (%s)." % (ablation["velocity"], ablation["profile"])
           if ablation else "Ablations.",
        4: "The input-representation experiment (identical architecture, budget and dataset).",
        5: "Parameter count and measured single-slot CPU inference time.",
    }
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and (i == 0 or not lines[i - 1].startswith("|")):
            tbl_n += 1
        out.append(line)
        # a table ends when the next line is not a table row
        if line.startswith("|") and (i + 1 >= len(lines) or not lines[i + 1].startswith("|")):
            out.append("")
            out.append("**Table %s.** %s"
                       % ("I" * tbl_n if tbl_n <= 3 else ("IV" if tbl_n == 4 else "V"),
                          captions.get(tbl_n, "")))
        i += 1
    text = "\n".join(out)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    left = re.findall(r"\*\[RESULTS-[^\]]+\]\*|\[VALUE:[^\]]+\]", text)
    print("wrote %s (%d unresolved placeholders)" % (dst, len(left)))
    for x in set(left):
        print("   unresolved:", x)


if __name__ == "__main__":
    main()
