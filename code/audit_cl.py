"""Audit the IEEE Communications Letters version of the paper.

Checks what a submission could be rejected for: unresolved placeholders, missing or
mis-numbered figures and tables, references, length against the 5-page limit, and -- most
importantly -- that every headline number in the text matches the JSON the experiments produced,
including the paired-test effect sizes and their significance markers.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "results")

path = os.path.join(PAPER, "manuscript_CL_final.md")
text = open(path, encoding="utf-8").read()
body = text.split("## References")[0]
ok = True

print("=== IEEE CL version audit ===")
print("file        :", os.path.relpath(path, ROOT))
# Count words the way a length limit cares about them: strip the Markdown decoration (table
# pipes, emphasis markers, link syntax) first.  Counting raw whitespace-separated tokens inflates
# the total badly, because every "|" and "**" becomes a token of its own.
def _words(s):
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"[|#*`_]", " ", s)
    return len(re.findall(r"[A-Za-z][A-Za-z\-']+", s))

words_body = _words(body)
words_all = words_body + _words(text.split("## References")[-1])
print("words       : %d  (body %d, references %d)"
      % (words_all, words_body, words_all - words_body))

figs = sorted(set(re.findall(r"\*\*Fig\. (\d+)\.\*\*", text)), key=int)
figcite = sorted(set(re.findall(r"Fig\. (\d+)", body)), key=int)
tabs = sorted(set(re.findall(r"\*\*TABLE ([IVX]+)\*\*", text)))
tabcite = sorted(set(re.findall(r"Table ([IVX]+)", body)))
print("figures     : captions %s | cited %s" % (",".join(figs), ",".join(figcite)))
print("tables      : captions %s | cited %s" % (",".join(tabs), ",".join(tabcite)))
print("references  :", len(re.findall(r"^\[\d+\]", text, re.M)))
print("equations   :", len(re.findall(r"\\tag\{", text)))

if figs != figcite:
    print("  ! figure numbering mismatch")
    ok = False
for t in tabs:
    if t not in tabcite:
        print("  ! TABLE %s is never cited in the text" % t)
        ok = False
if re.search(r"\[VALUE:|RESULTS-|PLACEHOLDER|TODO", text):
    print("  ! unresolved placeholder")
    ok = False
if "Fig. 4" in body or "TABLE IV" in text:
    print("  ! reference to a figure/table that does not exist")
    ok = False

# ---------------- numbers against the JSON
main = json.load(open(os.path.join(RESULTS, "eval_main.json")))
repf = json.load(open(os.path.join(RESULTS, "exp_representation_full.json")))
repq = json.load(open(os.path.join(RESULTS, "exp_input_representation.json")))
pld = json.load(open(os.path.join(RESULTS, "paired_Ld_full.json")))
pout = json.load(open(os.path.join(RESULTS, "paired_output.json")))
ptmp = json.load(open(os.path.join(RESULTS, "paired_temporal.json")))


def g(d, k):
    return d[str(k)] if str(k) in d else d[k]


def at(paid, key, snr):
    return paid["comparisons"][str(snr)][key]


print("\n=== observation-model numbers ===")
for lab, val, tol in (("11.9", repf["raw_nmse_db"] - repf["derotated_nmse_db"], 0.1),
                      ("19.2", repq["raw_nmse_db"] - repq["derotated_nmse_db"], 0.1)):
    present = lab in text
    good = present and abs(float(lab) - val) <= tol
    print("  %-6s quoted=%-4s measured=%+.2f  %s"
          % (lab, "yes" if present else "NO", val, "ok" if good else "CHECK"))
    ok = ok and good

print("\n=== propagation numbers (eval_main) ===")
for lab, val in (("-36.70", g(g(main["nmse_all"]["TDL-C"], 3.0), 25.0)["DFT-denoise"]),
                 ("-18.75", g(g(main["nmse_all"]["TDL-C"], 30.0), 25.0)["DFT-denoise"]),
                 ("-3.55", g(g(main["nmse_all"]["TDL-C"], 200.0), 25.0)["DFT-denoise"]),
                 ("-10.54", g(g(main["nmse_all"]["TDL-C"], 100.0), 25.0)["ChannelNet"])):
    present = lab in text
    good = present and abs(float(lab) - val) <= 0.02
    print("  %-8s quoted=%-4s measured=%+.2f  %s"
          % (lab, "yes" if present else "NO", val, "ok" if good else "CHECK"))
    ok = ok and good

print("\n=== paired-test effect sizes quoted in the text ===")
checks = [
    ("0.63", at(pld, "Ld=2 - Ld=16", 20.0), "Ld=2 vs 16 at 20 dB"),
    ("0.59", at(pld, "Ld=4 - Ld=16", 25.0), "Ld=4 vs 16 at 25 dB"),
    ("0.31", at(pld, "Ld=2 - Ld=16", -5.0), "Ld=2 deficit at -5 dB"),
    ("2.30", at(pout, "out-Ld2 - feat-Ld2", 25.0), "output-constraint penalty at 25 dB"),
    ("1.92", at(pout, "out-Ld2 - feat-Ld2", 20.0), "output-constraint penalty at 20 dB"),
    ("4.26", at(ptmp, "M=8 - M=2", 25.0), "M=8 penalty at 25 dB"),
    ("1.43", at(ptmp, "M=8 - M=2", 0.0), "M=8 penalty at 0 dB"),
]
for lab, rec, what in checks:
    val = rec["mean_db"]
    present = lab in text
    good = present and abs(float(lab) - abs(val)) <= 0.02
    print("  %-6s %-34s measured=%+.2f  sig=%-5s  %s"
          % (lab, what, val, rec["significant"], "ok" if good else "CHECK"))
    ok = ok and good

print("\n=== significance annotation ===")
n_star = len(re.findall(r"\|\s*[+-]?\d+\.\d+\*", text))
print("  note in caption          :", "$^{*}$ marks" in text)
print("  starred cells in tables  :", n_star)
if "$^{*}$ marks" not in text or n_star < 15:
    print("  ! significance annotation looks incomplete")
    ok = False

print("\n=== length ===")
pages = words_all / 1100.0 + len(figs) * 0.22 + len(tabs) * 0.15
print("  ~%.1f pages of 5 (body %.1f + floats %.1f)"
      % (pages, words_all / 1100.0, len(figs) * 0.22 + len(tabs) * 0.15))
if pages > 5.0:
    print("  ! likely over the 5-page limit")
    ok = False

print("\nRESULT:", "PASS" if ok else "NEEDS ATTENTION")
