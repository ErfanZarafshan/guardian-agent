# Building `report.pdf` from the LaTeX source

The full report source lives in the `docs/` directory:

```
docs/
├── report.tex          # main document
├── refs.bib            # bibliography
├── icml2026.sty        # ICML 2026 style file
├── icml2026.bst        # ICML 2026 BibTeX style
├── algorithm.sty       # for any pseudocode (not currently used)
├── algorithmic.sty     # for any pseudocode (not currently used)
├── fancyhdr.sty        # for headers/footers
└── report.pdf          # the compiled output
```

You have **two options** for building. Either produces an identical PDF.

---

## Option A — Overleaf (recommended, no install)

Overleaf is the standard online LaTeX editor and is what the assignment
documentation references explicitly.

1. Go to <https://www.overleaf.com> and sign up / sign in
2. Click **New Project → Upload Project**
3. Make a `.zip` of the `docs/` folder contents (everything in there)
4. Upload the zip
5. Overleaf will detect `report.tex` as the main document
6. Click **Recompile**
7. The PDF appears on the right; **Download PDF** from the top-right menu

---

## Option B — Local (Mac with MacTeX or Linux with TeX Live)

```bash
# From the repo root:
cd docs

# 1. First pass: lay out the document
pdflatex report.tex

# 2. Resolve bibliography
bibtex report

# 3. Two more passes to settle cross-references and citations
pdflatex report.tex
pdflatex report.tex

# Output:
ls -la report.pdf
```

If you do not have LaTeX installed on your Mac, the easiest option is to
install **MacTeX**:

```bash
brew install --cask mactex-no-gui
```

(That's a 4 GB download; the GUI version is even larger but adds the
TeXShop editor. The `no-gui` package is enough for `pdflatex` and
`bibtex` from the command line.)

---

## Editing tips

- The author block is near the top of `report.tex` — look for
  `\begin{icmlauthorlist}`. Update names/affiliations there if needed
  (currently set to Erfan Zarafshan + Maby Gavilan Abanto, both at LSU).
- `\usepackage[accepted]{icml2026}` is set, which means the
  blind-review redaction is OFF and your names will appear on the title
  page. Do **not** remove `[accepted]` for your final submission.
- The page count is dictated by `\twocolumn[ ... ]`. Don't change the
  document class or the column setup — the rubric checks formatting.
- The `\icmlcorrespondingauthor` macro produces the footer-line email.
  Currently set to `ezaraf1@lsu.edu`; change if needed.

---

## Troubleshooting

**"File `icml2026.sty' not found"** — make sure you're running
`pdflatex` from inside `docs/` so it can find the local `.sty` files.

**"Citation `Foo' on page X undefined"** — you forgot to run `bibtex`.
Run the four-command sequence above in order: `pdflatex`, `bibtex`,
`pdflatex`, `pdflatex`.

**Page count over 6** — fix in `report.tex` by trimming text. The
methodology and results sections are the most compressible. Keep the
abstract and the contributions list intact.
