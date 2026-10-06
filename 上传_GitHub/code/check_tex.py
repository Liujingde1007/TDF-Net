"""Structural validation of the IEEEtran submission, without needing a LaTeX installation.

Checks what a compiler would report as an error (unbalanced environments, undefined citation keys,
missing figure files, leftover placeholders) and what the journal would reject on sight (wrong
document class, no biography, over-length).  It is not a substitute for compiling, but it catches
the mistakes that are actually likely here.
"""
from __future__ import annotations

import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "TDF-Net-ieeeCL.tex")
FIGS = os.path.join(ROOT, "figures")

text = io.open(TEX, encoding="utf-8").read()
ok = True

print("=== structural check: %s ===" % os.path.relpath(TEX, ROOT))

if not re.search(r"\\documentclass\[journal\]\{IEEEtran\}", text):
    print("  ! documentclass is not IEEEtran journal"); ok = False
else:
    print("  documentclass          : IEEEtran [journal]")

for env in ("document", "abstract", "IEEEkeywords", "thebibliography", "IEEEbiographynophoto"):
    b = len(re.findall(r"\\begin\{%s\}" % env, text))
    e = len(re.findall(r"\\end\{%s\}" % env, text))
    if b != e or b == 0:
        print("  ! environment %-22s begin=%d end=%d" % (env, b, e)); ok = False
    else:
        print("  environment %-22s : %d balanced" % (env, b))

counts = {
    "equations": r"\\begin\{equation\}",
    "figures": r"\\begin\{figure",
    "tables": r"\\begin\{table",
    "bibitems": r"\\bibitem",
}
for name, pat in counts.items():
    print("  %-22s : %d" % (name, len(re.findall(pat, text))))

# every \cite key must have a matching \bibitem
keys = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
used = set()
for m in re.findall(r"\\cite\{([^}]+)\}", text):
    used.update(k.strip() for k in m.split(","))
missing = used - keys
unused = keys - used
print("  citation keys          : %d used, %d defined, %d unresolved, %d never cited"
      % (len(used), len(keys), len(missing), len(unused)))
if missing:
    print("  ! UNRESOLVED CITES: %s" % ", ".join(sorted(missing))); ok = False
if unused:
    print("    (defined but never cited: %s)" % ", ".join(sorted(unused)))

# every included figure must exist on disk.  The name may be written with or without an
# extension, so accept the literal path first and then the .pdf/.png alternatives.
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", text)
fig_ok = True
for f in figs:
    base, ext = os.path.splitext(f)
    cands = [os.path.join(FIGS, f)]
    if ext == "":
        cands += [os.path.join(FIGS, base + e) for e in (".pdf", ".png")]
    hits = [os.path.exists(c) for c in cands]
    if not any(hits):
        print("  ! missing figure file: %s" % f)
        for c, h in zip(cands, hits):
            print("      candidate %r exists=%s" % (os.path.basename(c), h))
        fig_ok = False
        ok = False
if fig_ok:
    print("  figure files           : %d referenced, all present" % len(figs))

# labels and references
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
undef = refs - labels
print("  labels / refs          : %d labels, %d refs, %d undefined" % (len(labels), len(refs), len(undef)))
if undef:
    print("  ! UNDEFINED REFS: %s" % ", ".join(sorted(undef))); ok = False

# placeholders and journal requirements
ph = re.findall(r"%<TABLE_|\[VALUE:|\[FAMILY\]|\[email\]|TODO", text)
print("  unresolved placeholders: %d" % len(ph))
if ph:
    print("  ! placeholders remain"); ok = False
if "\\begin{IEEEbiographynophoto}" not in text:
    print("  ! no author biography (IEEE CL requires one)"); ok = False
else:
    print("  author biography       : present")

# Table width.  A tabular wider than the text block overflows into the facing column, which no
# other check here can see.  Absolute widths cannot be predicted from the source: they depend on
# the resolved font metrics and on how LaTeX breaks the headers.  What *can* be stated reliably is
# relative width, and the structural rule that a many-column table must span both columns.
def col_widths(tab):
    rows = [r for r in tab.split("\\\\") if "&" in r]
    ncols = max(len(r.split("&")) for r in rows) if rows else 0
    w = [0] * ncols
    for r in rows:
        for i, cell in enumerate(r.split("&")):
            cell = re.sub(r"\\(?:cite|textbf|emph|SI)\{?", "", cell)
            cell = re.sub(r"[{}$\\]", "", cell)
            w[i] = max(w[i], len(cell.strip()))
    return w


print("  table width (relative; confirm against the compile log):")
widths = {}
for env, tb in re.findall(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", text, re.S):
    if "\\begin{tabular}" not in tb:
        continue
    w = col_widths(tb[tb.index("\\begin{tabular}"):])
    total = sum(w) + 3 * max(0, len(w) - 1)
    span = 2 if env == "table*" else 1
    lab = re.search(r"\\label\{([^}]+)\}", tb)
    name = lab.group(1) if lab else "?"
    widths[name] = (span, len(w), total)
    print("      %-11s span=%d cols=%2d natural width %3d" % (name, span, len(w), total))

# A table with more than five columns cannot fit an IEEEtran single column whatever the font, so
# it must be a table*.  This is a hard rule, not an estimate.
for name, (span, ncols, total) in widths.items():
    if span == 1 and ncols > 5:
        print("  ! %s has %d columns but does not span both columns; it will overflow"
              % (name, ncols))
        ok = False
if widths:
    widest = max(widths, key=lambda k: widths[k][2] / widths[k][0])
    print("      widest per-column: %s -- if the log reports an Overfull \\hbox in a table, "
          "shorten this one first" % widest)

# length estimate from the source (words + float allowance)
body = text.split("\\begin{thebibliography}")[0]
words = len(re.findall(r"[A-Za-z][A-Za-z\-']+", body))
n_fig = len(re.findall(r"\\begin\{figure", text))
n_tab = len(re.findall(r"\\begin\{table", text))
pages = words / 1100.0 + n_fig * 0.22 + n_tab * 0.15
print("  body words             : %d" % words)
print("  floats                 : %d figures, %d tables" % (n_fig, n_tab))
print("  estimated length       : %.1f pages of 5" % pages)
if pages > 5.0:
    print("  ! likely over the 5-page limit"); ok = False

# spot-check the prose numbers that come from the main evaluation
main = json.load(open(os.path.join(ROOT, "results", "eval_main.json")))
prof = main["nmse_all"]["TDL-C"]


def cell(vel, snr, method):
    """eval_main.json nests nmse_all[profile][velocity_str][snr_str][method]."""
    for kv in (str(vel), vel):
        if kv in prof:
            by_snr = prof[kv]
            break
    else:
        raise KeyError("velocity %r not in eval_main (have %s)" % (vel, list(prof)))
    for ks in (str(snr), snr):
        if ks in by_snr:
            return by_snr[ks][method]
    raise KeyError("SNR %r not present under velocity %r" % (snr, vel))


print("\n=== prose numbers vs eval_main.json ===")
for vel, snr, method, quoted in ((100.0, 0.0, "OMP", "-7.70"),
                                 (3.0, 25.0, "DFT-denoise", "-36.70"),
                                 (30.0, 25.0, "DFT-denoise", "-18.75"),
                                 (200.0, 25.0, "DFT-denoise", "-3.55"),
                                 (100.0, 25.0, "ChannelNet", "-10.54")):
    val = cell(vel, snr, method)
    good = abs(float(quoted) - val) <= 0.02 and quoted in text
    print("  %-8s %-12s at %6.1f km/h, %+5.1f dB : measured %+.2f  %s"
          % (quoted, method, vel, snr, val, "ok" if good else "CHECK"))
    if not good:
        ok = False

print("\nRESULT:", "PASS" if ok else "NEEDS ATTENTION")
