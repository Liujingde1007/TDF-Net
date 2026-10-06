"""Publication-quality figures for the manuscript.

Every figure is written to ``figures/`` as vector PDF (for LaTeX) and 600-dpi PNG (for Word and
for journals that require raster figures).  All data comes from the JSON produced by ``eval.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
    # embed TrueType (Type 42) rather than Type 3 fonts: required by IEEE and most publishers
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})

# consistent style per estimator, proposed method last so it sits on top
STYLE = {
    "LS-linear":   dict(color="#7f7f7f", marker="v", ls="--"),
    "DFT-denoise": dict(color="#1f77b4", marker="s", ls="-."),
    "OMP":         dict(color="#2ca02c", marker="^", ls=":"),
    "LMMSE":       dict(color="#9467bd", marker="D", ls="--"),
    "ChannelNet":  dict(color="#ff7f0e", marker="o", ls="-"),
    "CDRN+GRU":    dict(color="#d62728", marker="P", ls="-"),
    "TDF-Net":     dict(color="#000000", marker="*", ls="-", lw=2.0, ms=8),
}


def style(name):
    return STYLE.get(name, dict(color="k", ls="-"))


def save(fig, name: str, formats=("pdf", "png")):
    os.makedirs(FIGS, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(FIGS, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote figures/%s.%s" % (name, "|".join(formats)))


def get(d: dict, key):
    return d[str(key)] if str(key) in d else d[key]


# ---------------------------------------------------------------------------------------
def fig_nmse(payload, profile, velocity, name):
    snrs = payload["snr_db"]
    data = get(payload["nmse_all"][profile], velocity)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for m in payload["methods"]:
        if m not in get(data, snrs[0]):
            continue
        y = [get(data, s)[m] for s in snrs]
        st = style(m)
        ax.plot(snrs, y, label=m, **st)
    ax.set_xlabel(r"SNR per resource element [dB]")
    ax.set_ylabel(r"NMSE [dB]")
    ax.set_title(r"%s, $v = %g$ km/h" % (profile, velocity))
    ax.legend(framealpha=0.9, loc="best")
    save(fig, name)


def fig_nmse_vs_velocity(payload, profile, snr, name):
    vels = payload["velocities"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for m in payload["methods"]:
        y = []
        for v in vels:
            entry = get(payload["nmse_all"][profile], v)
            y.append(get(entry, snr)[m])
        ax.plot(vels, y, label=m, **style(m))
    ax.set_xlabel(r"UE velocity [km/h]")
    ax.set_ylabel(r"NMSE [dB]")
    ax.set_title(r"%s, SNR $= %g$ dB" % (profile, snr))
    ax.legend(framealpha=0.9, loc="best")
    save(fig, name)


def fig_ber(payload, velocity, name):
    snrs = payload["snr_db"]
    data = get(payload["ber"], velocity)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for m in payload["methods"]:
        if m not in get(data, snrs[0]):
            continue
        y = [max(get(data, s)[m], 1e-6) for s in snrs]
        ax.semilogy(snrs, y, label=m, **style(m))
    ax.set_xlabel(r"SNR per resource element [dB]")
    ax.set_ylabel(r"BER (QPSK, 1-tap MMSE equaliser)")
    ax.set_title(r"TDL-C, $v = %g$ km/h" % velocity)
    ax.legend(framealpha=0.9, loc="best")
    save(fig, name)


def fig_ablation(payload, name):
    """NMSE vs temporal window M, and the delay-branch ablation."""
    abl = payload.get("ablation")
    if not abl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    snrs = payload["snr_db"]

    ax = axes[0]
    ms = sorted(int(k) for k in abl["by_M"].keys())
    for m in ms:
        y = abl["by_M"][str(m)]
        ax.plot(snrs[:len(y)], y, marker="o", label=r"$M=%d$" % m)
    ax.set_xlabel(r"SNR [dB]")
    ax.set_ylabel(r"NMSE [dB]")
    ax.set_title(r"(a) temporal window $M$")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    for key, lab in (("with_delay", r"with delay branch"),
                     ("no_delay", r"without delay branch")):
        if key not in abl:
            continue
        y = abl[key]
        ax.plot(snrs[:len(y)], y, marker="s", label=lab)
    ax.set_xlabel(r"SNR [dB]")
    ax.set_ylabel(r"NMSE [dB]")
    ax.set_title(r"(b) delay-domain branch")
    ax.legend(framealpha=0.9)
    save(fig, name)


def fig_complexity(payload, name):
    comp = payload.get("complexity")
    if not comp:
        return
    names, pars, times = [], [], []
    for m in payload["methods"]:
        if m in comp and comp[m]["params"]:
            names.append(m)
            pars.append(comp[m]["params"])
            times.append(comp[m]["inference_ms_per_slot"])
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].barh(names, pars, color=[style(n)["color"] for n in names])
    axes[0].set_xlabel(r"trainable parameters")
    axes[0].set_title(r"(a) model size")
    axes[1].barh(names, times, color=[style(n)["color"] for n in names])
    axes[1].set_xlabel(r"CPU inference time per slot [ms]")
    axes[1].set_title(r"(b) inference cost")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    save(fig, name)


def fig_training_curves(tags, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for tag, lab in tags:
        path = os.path.join(RESULTS, "history_%s.json" % tag)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            h = json.load(f)
        st = style(lab)
        axes[0].plot(h["epochs"], 10 * np.log10(h["train_loss"]), label=lab, **st)
        axes[1].plot(h["epochs"], 10 * np.log10(h["val_loss"]), label=lab, **st)
    for ax, t in zip(axes, (r"(a) training loss", r"(b) validation loss")):
        ax.set_xlabel(r"epoch")
        ax.set_ylabel(r"NMSE [dB]")
        ax.set_title(t)
        ax.legend(framealpha=0.9)
    save(fig, name)


def fig_delay_interpretation(payload, name):
    """Learned delay-domain window of the proposed model, if exported."""
    path = os.path.join(RESULTS, "delay_window.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    w = np.array(d["window"])
    taps = np.array(d["active_taps"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].stem(np.arange(len(w)), w)
    axes[0].set_xlabel(r"delay tap index $d$")
    axes[0].set_ylabel(r"learned window $w_d$")
    axes[0].set_title(r"(a) learned delay-domain window")
    axes[1].stem(taps[0], taps[1])
    axes[1].set_xlabel(r"delay tap index $d$")
    axes[1].set_ylabel(r"normalised tap power")
    axes[1].set_title(r"(b) true channel delay profile")
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval_main")
    ap.add_argument("--ablation", default="eval_ablation")
    args = ap.parse_args()

    with open(os.path.join(RESULTS, args.inp + ".json")) as f:
        payload = json.load(f)
    payload.setdefault("methods", ["LS-linear", "DFT-denoise", "OMP", "LMMSE",
                                   "ChannelNet", "CDRN+GRU", "TDF-Net"])
    print("figures:")
    p0 = payload["profiles"][0]
    v0 = payload["velocities"][1] if len(payload["velocities"]) > 1 else payload["velocities"][0]
    fig_nmse(payload, p0, v0, "fig_nmse_%s_v%g" % (p0.replace("-", "").lower(), v0))
    fig_nmse_vs_velocity(payload, p0, payload["snr_db"][-1], "fig_nmse_vs_velocity")
    if "ber" in payload:
        for v in payload["ber"]:
            fig_ber(payload, float(v), "fig_ber_v%g" % float(v))
    fig_complexity(payload, "fig_complexity")
    if len(payload["profiles"]) > 1:
        fig_nmse(payload, payload["profiles"][1], 100.0, "fig_nmse_profile_%s"
                 % payload["profiles"][1].replace("-", "").lower())

    abl_path = os.path.join(RESULTS, args.ablation + ".json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            payload["ablation"] = json.load(f)
        fig_ablation(payload, "fig_ablation")
    fig_training_curves([("tdfnet_M4", "TDF-Net"), ("cdrn_M4", "CDRN+GRU"),
                         ("channelnet", "ChannelNet")], "fig_training")
    fig_delay_interpretation(payload, "fig_delay_window")


if __name__ == "__main__":
    main()
