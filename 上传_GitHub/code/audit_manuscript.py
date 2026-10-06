"""Final audit of the generated manuscript: placeholders, cross-references, quoted numbers."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "results")

path = os.path.join(PAPER, "manuscript_illustrated.md")
text = open(path, encoding="utf-8").read()

print("=== manuscript ===")
print("title      :", text.split("\n")[0].lstrip("# ").strip())
print("words      :", len(text.split()))
print("placeholders unresolved:",
      len(re.findall(r"\[VALUE:|RESULTS-|PLACEHOLDER|TODO|XXX", text)))
figs = sorted(set(re.findall(r"Fig\. (\d+)", text)), key=int)
caps = sorted(set(re.findall(r"\*\*Fig\. (\d+)\.\*\*", text)), key=int)
print("figures cited   :", ",".join(figs))
print("figure captions :", ",".join(caps))
tabs = sorted(set(re.findall(r"Table ([IVX]+)", text)))
tcaps = sorted(set(re.findall(r"\*\*Table ([IVX]+)\.\*\*", text)))
print("tables cited    :", ",".join(tabs))
print("table captions  :", ",".join(tcaps))
print("references      :", len(re.findall(r"^\[\d+\]", text, re.M)))

print("\n=== numbers quoted in the text vs measured ===")
checks = {
    "753 298": "TDF-Net parameters",
    "257 762": "ResCNN parameters",
    "373 410": "ResCNN+GRU parameters",
    "786 322": "LSTM parameters",
    "14 114": "ChannelNet parameters",
    "65 808": "delay branch parameters",
    "131 968": "temporal branch parameters",
    "3000 sequences": "training set size",
}
for needle, what in checks.items():
    n = text.count(needle)
    print("  %-18s %-28s occurrences=%d" % (needle, what, n))

stale = {"753 586": "old parameter count", "144 000": "old delay-branch count",
         "197 000": "old temporal count", "2500 sequences": "old training size",
         "[VALUE:": "placeholder"}
print("\n=== stale values ===")
bad = 0
for needle, what in stale.items():
    n = text.count(needle)
    if n:
        print("  STALE: %-18s (%s) occurrences=%d" % (needle, what, n))
        bad += n
print("  none" if not bad else "  %d stale occurrences" % bad)

print("\n=== measured values for cross-check ===")
rep = json.load(open(os.path.join(RESULTS, "exp_input_representation.json")))
repf = json.load(open(os.path.join(RESULTS, "exp_representation_full.json")))
abl = json.load(open(os.path.join(RESULTS, "eval_ablation.json")))
main = json.load(open(os.path.join(RESULTS, "eval_main.json")))
print("  de-rotation gap, full protocol : %.2f dB" % (repf["raw_nmse_db"] - repf["derotated_nmse_db"]))
print("  de-rotation gap, fixed case    : %.2f dB" % (rep["raw_nmse_db"] - rep["derotated_nmse_db"]))
d = [abl["with_delay"][i] - abl["no_delay"][i] for i in range(len(abl["with_delay"]))]
print("  delay-branch effect            : mean %.2f dB, max %.2f dB" % (
    sum(abs(x) for x in d) / len(d), max(abs(x) for x in d)))
print("  M=4 vs M=2 at 25 dB            : %+.2f dB" % (abl["by_M"]["4"][-1] - abl["by_M"]["2"][-1]))
print("  M=8 vs M=2 at 25 dB            : %+.2f dB" % (abl["by_M"]["8"][-1] - abl["by_M"]["2"][-1]))
r = main["nmse_all"]["TDL-C"]["100.0"]["25.0"]
print("  ChannelNet floor at 25 dB      : %.2f dB" % r["ChannelNet"])
print("  200 km/h lead over ResCNN (25 dB): %+.2f dB" % (
    main["nmse_all"]["TDL-C"]["200.0"]["25.0"]["ResCNN"]
    - main["nmse_all"]["TDL-C"]["200.0"]["25.0"]["TDF-Net"]))
