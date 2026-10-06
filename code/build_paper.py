"""Build the manuscript deliverables.

  * ``paper/TDF-Net-manuscript.docx``  -- Word file with native (OMML) equations, styled with
    an IEEE-like reference template.
  * ``paper/TDF-Net-manuscript.tex``   -- LaTeX version for an IEEEtran submission.

Figures referenced from ``figures/`` are embedded into the Word file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pypandoc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
FIGS = os.path.join(ROOT, "figures")


def make_reference_docx(path: str):
    """Create a pandoc reference document and patch the main styles towards IEEE look."""
    if os.path.exists(path):
        return
    pypandoc.convert_text("# T\n\nbody\n", "docx", format="markdown",
                          outputfile=path, extra_args=["--standalone"])
    # patch styles.xml
    import zipfile
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                xml = data.decode("utf-8")
                # body font -> Times New Roman 10pt (20 half-points)
                xml = xml.replace('w:asciiTheme="minorHAnsi"', 'w:ascii="Times New Roman"')
                xml = xml.replace('w:ascii="Calibri"', 'w:ascii="Times New Roman"')
                xml = xml.replace('w:cs="Calibri"', 'w:cs="Times New Roman"')
                xml = xml.replace('w:hAnsi="Calibri"', 'w:hAnsi="Times New Roman"')
                xml = xml.replace('w:val="22"', 'w:val="20"')
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    print("  reference template: %s" % path)


def convert_docx(md_path: str, out_path: str, ref: str):
    extra = ["--standalone", "--toc", "--toc-depth=2", "--number-sections"]
    if os.path.exists(ref):
        extra += ["--reference-doc=" + ref]
    pypandoc.convert_file(md_path, "docx", outputfile=out_path, format="markdown",
                          extra_args=extra)
    print("  wrote %s" % out_path)


def convert_tex(md_path: str, out_path: str):
    pypandoc.convert_file(md_path, "latex", outputfile=out_path, format="markdown",
                          extra_args=["--standalone"])
    print("  wrote %s" % out_path)


def main():
    md = os.path.join(PAPER, "manuscript_illustrated.md")
    if not os.path.exists(md):
        md = os.path.join(PAPER, "manuscript_final.md")
    if not os.path.exists(md):
        md = os.path.join(PAPER, "manuscript.md")
    if not os.path.exists(md):
        sys.exit("no manuscript markdown found in %s" % PAPER)
    print("  source: %s" % os.path.relpath(md, ROOT))
    ref = os.path.join(PAPER, "reference.docx")
    make_reference_docx(ref)
    convert_docx(md, os.path.join(PAPER, "TDF-Net-manuscript.docx"), ref)
    convert_tex(md, os.path.join(PAPER, "TDF-Net-manuscript.tex"))


if __name__ == "__main__":
    main()
